package coordinator

import (
	"context"
	"fmt"
	"strings"

	db "github.com/jackc/pgx/v4"
	"go.uber.org/zap"

	"github.com/featureform/serving/metadata"
	provider "github.com/featureform/serving/provider"
	runner "github.com/featureform/serving/runner"
	clientv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/client/v3/concurrency"
)

func templateReplace(template string, replacements map[string]string) (string, error) {
	fmt.Println("Template", template)
	formattedString := ""
	numEscapes := strings.Count(template, "{{")
	for i := 0; i < numEscapes; i++ {
		split := strings.SplitN(template, "{{", 2)
		afterSplit := strings.SplitN(split[1], "}}", 2)
		key := strings.TrimSpace(afterSplit[0])
		replacement, has := replacements[key]
		if !has {
			return "", fmt.Errorf("No key set")
		}
		formattedString += fmt.Sprintf("%s%s", split[0], replacement)
		template = afterSplit[1]
	}
	formattedString += template
	fmt.Println("Formatted", formattedString)
	return formattedString, nil
}

type Coordinator struct {
	Metadata   *metadata.Client
	Logger     *zap.SugaredLogger
	EtcdClient *clientv3.Client
	KVClient   *clientv3.KV
	Spawner    JobSpawner
}

type JobSpawner interface {
	GetJobRunner(jobName string, config runner.Config) (runner.Runner, error)
}

type KubernetesJobSpawner struct{}

type MemoryJobSpawner struct{}

func GetLockKey(jobKey string) string {
	return fmt.Sprintf("LOCK_%s", jobKey)
}

func (k *KubernetesJobSpawner) GetJobRunner(jobName string, config runner.Config) (runner.Runner, error) {
	kubeConfig := runner.KubernetesRunnerConfig{
		EnvVars:  map[string]string{"NAME": runner.CREATE_TRAINING_SET, "CONFIG": string(config)},
		Image:    "featureform/worker",
		NumTasks: 1,
	}
	jobRunner, err := runner.NewKubernetesRunner(kubeConfig)
	if err != nil {
		return nil, err
	}
	return jobRunner, nil
}

func (k *MemoryJobSpawner) GetJobRunner(jobName string, config runner.Config) (runner.Runner, error) {
	jobRunner, err := runner.Create(jobName, config)
	if err != nil {
		return nil, err
	}
	return jobRunner, nil
}

func NewCoordinator(meta *metadata.Client, logger *zap.SugaredLogger, cli *clientv3.Client, spawner JobSpawner) (*Coordinator, error) {
	logger.Debug("Creating new coordinator")
	kvc := clientv3.NewKV(cli)
	return &Coordinator{
		Metadata:   meta,
		Logger:     logger,
		EtcdClient: cli,
		KVClient:   &kvc,
		Spawner:    spawner,
	}, nil
}

const MAX_ATTEMPTS = 1

func (c *Coordinator) WatchForNewJobs() error {
	getResp, err := (*c.KVClient).Get(context.Background(), "JOB_", clientv3.WithPrefix())
	if err != nil {
		return err
	}
	for _, kv := range getResp.Kvs {
		fmt.Println("FOUND:", kv)
		go func() {
			err := c.executeJob(string(kv.Key))
			if err != nil {
				fmt.Println(err)
			}
		}()
	}
	for {
		fmt.Println("WATCHING")
		rch := c.EtcdClient.Watch(context.Background(), "JOB_", clientv3.WithPrefix())
		for wresp := range rch {
			fmt.Println("FOUND:", wresp)
			for _, ev := range wresp.Events {
				fmt.Println("FOUND:", ev)
				if ev.Type == 0 {
					fmt.Println("EXECUTING:", ev.Kv.Key)
					go func() {
						err := c.executeJob(string(ev.Kv.Key))
						if err != nil {
							fmt.Println(err)
						}
					}()
				}

			}
		}
	}
	return nil
}

func (c *Coordinator) mapNameVariantsToTables(sources []metadata.NameVariant) (map[string]string, error) {
	sourceMap := make(map[string]string)

	c.Logger.Debug(fmt.Sprintf("%v", sources))

	for _, nameVariant := range sources {
		var tableName string
		source, err := c.Metadata.GetSourceVariant(context.Background(), nameVariant)
		if err != nil {
			return nil, err
		}
		if source.Status() != metadata.READY {
			return nil, fmt.Errorf("source in query not ready")
		}
		providerResourceID := provider.ResourceID{Name: source.Name(), Variant: source.Variant()}
		if source.IsSQLTransformation() {
			tableName = provider.GetTransformationName(providerResourceID)
		} else if source.IsPrimaryDataSQLTable() {
			tableName = provider.GetPrimaryTableName(providerResourceID)
		}

		c.Logger.Debug("mapping name variants to table names")
		c.Logger.Debug(fmt.Sprintf("%v", nameVariant))

		c.Logger.Debug(tableName)

		sourceMap[fmt.Sprintf("%s.%s", nameVariant.Name, nameVariant.Variant)] = tableName
	}
	return sourceMap, nil
}

func sanitize(ident string) string {
	return db.Identifier{ident}.Sanitize()
}

func (c *Coordinator) runSQLTransformationJob(transformSource *metadata.SourceVariant, resID metadata.ResourceID, offlineStore provider.OfflineStore) error {
	templateString := transformSource.SQLTransformationQuery()
	sources := transformSource.SQLTransformationSources()
	c.Logger.Debug("retrieved sources from metadata")
	c.Logger.Debug(fmt.Sprintf("%v", sources))
	allReady := false
	for sourceVariants, err := c.Metadata.GetSourceVariants(context.Background(), sources); !allReady; sourceVariants, err = c.Metadata.GetSourceVariants(context.Background(), sources) {
		if err != nil {
			return err
		}
		total := len(sourceVariants)
		totalReady := 0
		for _, sourceVariant := range sourceVariants {
			if sourceVariant.Status() == metadata.READY {
				totalReady += 1
			}
			if sourceVariant.Status() == metadata.FAILED {
				return err
			}
		}
		if total == totalReady {
			allReady = true
		}
	}

	sourceMap, err := c.mapNameVariantsToTables(sources)
	if err != nil {
		return err
	}
	query, err := templateReplace(templateString, sourceMap)
	if err != nil {
		return err
	}
	providerResourceID := provider.ResourceID{Name: resID.Name, Variant: resID.Variant, Type: provider.Transformation}
	transformationConfig := provider.TransformationConfig{TargetTableID: providerResourceID, Query: query}
	c.Logger.Debug(fmt.Sprintf("%v", transformationConfig))
	if err := offlineStore.CreateTransformation(transformationConfig); err != nil {
		return err
	}
	if err := c.Metadata.SetStatus(context.Background(), resID, string(metadata.READY)); err != nil {
		return err
	}
	return nil
}

func (c *Coordinator) runPrimaryTableJob(transformSource *metadata.SourceVariant, resID metadata.ResourceID, offlineStore provider.OfflineStore) error {
	providerResourceID := provider.ResourceID{Name: resID.Name, Variant: resID.Variant}
	sourceName := transformSource.PrimaryDataSQLTableName()
	if sourceName == "" {
		return fmt.Errorf("no source name set")
	}
	if _, err := offlineStore.RegisterPrimaryFromSourceTable(providerResourceID, sourceName); err != nil {
		return err
	}
	if err := c.Metadata.SetStatus(context.Background(), resID, string(metadata.READY)); err != nil {
		return err
	}
	return nil
}

func (c *Coordinator) runRegisterSourceJob(resID metadata.ResourceID) error {
	source, err := c.Metadata.GetSourceVariant(context.Background(), metadata.NameVariant{resID.Name, resID.Variant})
	if err != nil {
		fmt.Println(err)
		return err
	}
	sourceProvider, err := source.FetchProvider(c.Metadata, context.Background())
	if err != nil {
		fmt.Println(err)
		return err
	}
	p, err := provider.Get(provider.Type(sourceProvider.Type()), sourceProvider.SerializedConfig())
	if err != nil {
		fmt.Println(err)
		return err
	}
	sourceStore, err := p.AsOfflineStore()
	if err != nil {
		fmt.Println(err)
		return err
	}
	if source.IsSQLTransformation() {
		return c.runSQLTransformationJob(source, resID, sourceStore)
	} else if source.IsPrimaryDataSQLTable() {
		return c.runPrimaryTableJob(source, resID, sourceStore)
	} else {
		return fmt.Errorf("source type not implemented")
	}
}

//should only be triggered when we are registering an ONLINE feature, not an offline one
func (c *Coordinator) runFeatureMaterializeJob(resID metadata.ResourceID) error {
	feature, err := c.Metadata.GetFeatureVariant(context.Background(), metadata.NameVariant{resID.Name, resID.Variant})
	if err != nil {
		return err
	}
	status := feature.Status()
	featureType := feature.Type()
	if status == metadata.READY || status == metadata.FAILED {
		return fmt.Errorf("feature already set to %s", metadata.ResourceStatus(status))
	}
	if err := c.Metadata.SetStatus(context.Background(), resID, string(metadata.PENDING)); err != nil {
		return err
	}
	sourceNameVariant := feature.Source()
	source, err := c.Metadata.GetSourceVariant(context.Background(), sourceNameVariant)
	if err != nil {
		fmt.Println(err)
		return err
	}
	sourceStatus := source.Status()
	if sourceStatus != metadata.READY {
		return fmt.Errorf("source of feature not ready")
	}
	sourceProvider, err := source.FetchProvider(c.Metadata, context.Background())
	if err != nil {
		fmt.Println(err)
		return err
	}
	p, err := provider.Get(provider.Type(sourceProvider.Type()), sourceProvider.SerializedConfig())
	if err != nil {
		fmt.Println(err)
		return err
	}
	sourceStore, err := p.AsOfflineStore()
	if err != nil {
		fmt.Println(err)
		return err
	}
	featureProvider, err := feature.FetchProvider(c.Metadata, context.Background())
	if err != nil {
		fmt.Println(err)
		return err
	}
	p, err = provider.Get(provider.Type(featureProvider.Type()), featureProvider.SerializedConfig())
	if err != nil {
		fmt.Println(err)
		return err
	}
	featureStore, err := p.AsOnlineStore()
	if err != nil {
		fmt.Println(err)
		return err
	}
	materializeRunner := runner.MaterializeRunner{
		Online:  featureStore,
		Offline: sourceStore,
		ID:      provider.ResourceID{Name: resID.Name, Variant: resID.Variant, Type: provider.Feature},
		VType:   provider.ValueType(featureType),
		Cloud:   runner.LocalMaterializeRunner,
	}
	completionWatcher, err := materializeRunner.Run()
	if err != nil {
		fmt.Println(err)
		return err
	}
	if err := completionWatcher.Wait(); err != nil {
		return err
	}
	if err := c.Metadata.SetStatus(context.Background(), resID, string(metadata.READY)); err != nil {
		return err
	}
	return nil
}

func (c *Coordinator) runTrainingSetJob(resID metadata.ResourceID) error {
	ts, err := c.Metadata.GetTrainingSetVariant(context.Background(), metadata.NameVariant{resID.Name, resID.Variant})
	if err != nil {
		fmt.Println(err)
		return err
	}
	status := ts.Status()
	if status == metadata.READY || status == metadata.FAILED {
		return fmt.Errorf("training Set already set to %s", metadata.ResourceStatus(status))
	}
	if err := c.Metadata.SetStatus(context.Background(), resID, string(metadata.PENDING)); err != nil {
		return err
	}
	providerEntry, err := ts.FetchProvider(c.Metadata, context.Background())
	if err != nil {
		fmt.Println(err)
		return err
	}
	p, err := provider.Get(provider.Type(providerEntry.Type()), providerEntry.SerializedConfig())
	if err != nil {
		fmt.Println(err)
		return err
	}
	store, err := p.AsOfflineStore()
	if err != nil {
		fmt.Println(err)
		return err
	}
	providerResID := provider.ResourceID{Name: resID.Name, Variant: resID.Variant, Type: provider.TrainingSet}
	if _, err := store.GetTrainingSet(providerResID); err == nil {
		return fmt.Errorf("training set already exists")
	}
	features := ts.Features()
	featureList := make([]provider.ResourceID, len(features))
	for i, feature := range features {
		featureList[i] = provider.ResourceID{Name: feature.Name, Variant: feature.Variant, Type: provider.Feature}
	}
	label, err := ts.FetchLabel(c.Metadata, context.Background())
	if err != nil {
		fmt.Println(err)
		return err
	}
	trainingSetDef := provider.TrainingSetDef{
		ID:       providerResID,
		Label:    provider.ResourceID{Name: label.Name(), Variant: label.Variant(), Type: provider.Label},
		Features: featureList,
	}
	tsRunnerConfig := runner.TrainingSetRunnerConfig{
		OfflineType:   provider.Type(providerEntry.Type()),
		OfflineConfig: providerEntry.SerializedConfig(),
		Def:           trainingSetDef,
	}
	serialized, err := tsRunnerConfig.Serialize()
	if err != nil {
		fmt.Println(err)
		return err
	}
	jobRunner, err := c.Spawner.GetJobRunner(runner.CREATE_TRAINING_SET, serialized)
	if err != nil {
		fmt.Println(err)
		return err
	}
	completionWatcher, err := jobRunner.Run()
	if err != nil {
		fmt.Println(err)
		return err
	}
	if err := completionWatcher.Wait(); err != nil {
		fmt.Println(err)
		return err
	}
	if err := c.Metadata.SetStatus(context.Background(), resID, string(metadata.READY)); err != nil {
		return err
	}
	return nil
}

func (c *Coordinator) getJob(mtx *concurrency.Mutex, key string) (*metadata.CoordinatorJob, error) {
	txn := (*c.KVClient).Txn(context.Background())
	response, err := txn.If(mtx.IsOwner()).Then(clientv3.OpGet(key)).Commit()
	if err != nil {
		return nil, fmt.Errorf("transaction did not succeed: %v", err)
	}
	isOwner := response.Succeeded //response.Succeeded sets true if transaction "if" statement true
	if !isOwner {
		return nil, fmt.Errorf("was not owner of lock")
	}
	responseData := response.Responses[0]
	responseKVs := responseData.GetResponseRange().GetKvs()
	if len(responseKVs) == 0 {
		return nil, fmt.Errorf("coordinator job does not exist")
	}
	responseValue := responseKVs[0].Value //Only single response for single key
	job := &metadata.CoordinatorJob{}
	if err := job.Deserialize(responseValue); err != nil {
		return nil, fmt.Errorf("could not deserialize coordinator job: %v", err)
	}
	return job, nil
}

func (c *Coordinator) incrementJobAttempts(mtx *concurrency.Mutex, job *metadata.CoordinatorJob, jobKey string) error {
	job.Attempts += 1
	serializedJob, err := job.Serialize()
	if err != nil {
		return fmt.Errorf("could not serialize coordinator job. %v", err)
	}
	txn := (*c.KVClient).Txn(context.Background())
	response, err := txn.If(mtx.IsOwner()).Then(clientv3.OpPut(jobKey, string(serializedJob))).Commit()
	if err != nil {
		return fmt.Errorf("could not set iterated coordinator job. %v", err)
	}
	isOwner := response.Succeeded //response.Succeeded sets true if transaction "if" statement true
	if !isOwner {
		return fmt.Errorf("was not owner of lock")
	}
	return nil
}

func (c *Coordinator) deleteJob(mtx *concurrency.Mutex, key string) error {
	txn := (*c.KVClient).Txn(context.Background())
	response, err := txn.If(mtx.IsOwner()).Then(clientv3.OpDelete(key)).Commit()
	if err != nil {
		return fmt.Errorf("delete job transaction failed: %v", err)
	}
	isOwner := response.Succeeded //response.Succeeded sets true if transaction "if" statement true
	if !isOwner {
		return fmt.Errorf("was not owner of lock")
	}
	responseData := response.Responses[0] //OpDelete always returns single response
	numDeleted := responseData.GetResponseDeleteRange().Deleted
	if numDeleted != 1 { //returns 0 if delete key did not exist
		return fmt.Errorf("job Already deleted")
	}
	return nil
}

func (c *Coordinator) hasJob(id metadata.ResourceID) (bool, error) {
	getResp, err := (*c.KVClient).Get(context.Background(), metadata.GetJobKey(id), clientv3.WithPrefix())
	if err != nil {
		return false, err
	}
	responseLength := len(getResp.Kvs)
	if responseLength > 0 {
		return true, nil
	}
	return false, nil
}

func (c *Coordinator) createJobLock(jobKey string, s *concurrency.Session) (*concurrency.Mutex, error) {
	mtx := concurrency.NewMutex(s, GetLockKey(jobKey))
	if err := mtx.Lock(context.Background()); err != nil {
		return nil, err
	}
	return mtx, nil
}

func (c *Coordinator) markJobFailed(job *metadata.CoordinatorJob) error {
	if err := c.Metadata.SetStatus(context.Background(), job.Resource, string(metadata.FAILED)); err != nil {
		return fmt.Errorf("could not set job status to failed: %v", err)
	}
	return nil
}

func (c *Coordinator) executeJob(jobKey string) error {
	s, err := concurrency.NewSession(c.EtcdClient, concurrency.WithTTL(1))
	if err != nil {

		return err
	}
	defer s.Close()
	mtx, err := c.createJobLock(jobKey, s)
	if err != nil {
		return err
	}
	defer func() {
		if err := mtx.Unlock(context.Background()); err != nil {
			fmt.Println(err)
			panic(err)
		}
	}()
	job, err := c.getJob(mtx, jobKey)
	if err != nil {
		return err
	}
	if job.Attempts > MAX_ATTEMPTS {
		return c.markJobFailed(job)
	}
	if err := c.incrementJobAttempts(mtx, job, jobKey); err != nil {
		return err
	}
	switch job.Resource.Type {
	case metadata.TRAINING_SET_VARIANT:
		if err := c.runTrainingSetJob(job.Resource); err != nil {
			statusErr := c.Metadata.SetStatus(context.Background(), job.Resource, fmt.Sprintf("%v", err))
			fmt.Println(err)
			return fmt.Errorf("training set job failed: %v", fmt.Sprintf("%v: %v", err, statusErr))
		}
	case metadata.FEATURE_VARIANT:
		if err := c.runFeatureMaterializeJob(job.Resource); err != nil {
			statusErr := c.Metadata.SetStatus(context.Background(), job.Resource, fmt.Sprintf("%v", err))
			fmt.Println(err)
			return fmt.Errorf("feature variant job failed: %v", fmt.Sprintf("%v: %v", err, statusErr))
		}
	case metadata.SOURCE_VARIANT:
		if err := c.runRegisterSourceJob(job.Resource); err != nil {
			statusErr := c.Metadata.SetStatus(context.Background(), job.Resource, fmt.Sprintf("%v", err))
			fmt.Println(err)
			return fmt.Errorf("source variant job failed: %v", fmt.Sprintf("%v: %v", err, statusErr))
		}
	default:
		return fmt.Errorf("not a valid resource type for running jobs")
	}
	if err := c.deleteJob(mtx, jobKey); err != nil {
		fmt.Println(err)
		return err
	}
	return nil
}
