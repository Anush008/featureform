package main

import (
	"context"
	"fmt"
	"google.golang.org/grpc/credentials/insecure"
	"io"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/featureform/serving/metadata"
	pb "github.com/featureform/serving/metadata/proto"
	"go.uber.org/zap"
	"google.golang.org/grpc"
)

type ApiServer struct {
	Logger     *zap.SugaredLogger
	address    string
	metaAddr   string
	meta       pb.MetadataClient
	metaClient *metadata.Client
	grpcServer *grpc.Server
	listener   net.Listener
	pb.UnimplementedApiServer
}

func NewApiServer(logger *zap.SugaredLogger, address string, metaAddr string) (*ApiServer, error) {
	return &ApiServer{
		Logger:   logger,
		address:  address,
		metaAddr: metaAddr,
	}, nil
}

func (serv *ApiServer) CreateUser(ctx context.Context, user *pb.User) (*pb.Empty, error) {
	serv.Logger.Infow("Creating User", "User", user.Name)
	return serv.meta.CreateUser(ctx, user)
}

func (serv *ApiServer) CreateProvider(ctx context.Context, provider *pb.Provider) (*pb.Empty, error) {
	serv.Logger.Infow("Creating Provider", "Provider", provider.Name)
	return serv.meta.CreateProvider(ctx, provider)
}

func (serv *ApiServer) CreateSourceVariant(ctx context.Context, source *pb.SourceVariant) (*pb.Empty, error) {
	serv.Logger.Infow("Creating Source Variant", "Source Variant", source.Name)
	switch casted := source.Definition.(type) {
	case *pb.SourceVariant_Transformation:
		transformation := casted.Transformation.Type.(*pb.Transformation_SQLTransformation).SQLTransformation
		qry := transformation.Query
		numEscapes := strings.Count(qry, "{{")
		sources := make([]*pb.NameVariant, numEscapes)
		for i := 0; i < numEscapes; i++ {
			split := strings.SplitN(qry, "{{", 2)
			afterSplit := strings.SplitN(split[1], "}}", 2)
			key := strings.TrimSpace(afterSplit[0])
			nameVariant := strings.SplitN(key, ".", 2)
			sources[i] = &pb.NameVariant{Name: nameVariant[0], Variant: nameVariant[1]}
			qry = afterSplit[1]
		}
		source.Definition.(*pb.SourceVariant_Transformation).Transformation.Type.(*pb.Transformation_SQLTransformation).SQLTransformation.Source = sources
	}
	return serv.meta.CreateSourceVariant(ctx, source)
}

func (serv *ApiServer) CreateEntity(ctx context.Context, entity *pb.Entity) (*pb.Empty, error) {
	serv.Logger.Infow("Creating Entity", "Entity", entity.Name)
	return serv.meta.CreateEntity(ctx, entity)
}

func (serv *ApiServer) CreateFeatureVariant(ctx context.Context, feature *pb.FeatureVariant) (*pb.Empty, error) {
	serv.Logger.Infow("Creating Feature Variant", "Feature Variant", feature.Name)
	return serv.meta.CreateFeatureVariant(ctx, feature)
}

func (serv *ApiServer) CreateLabelVariant(ctx context.Context, label *pb.LabelVariant) (*pb.Empty, error) {
	serv.Logger.Infow("Creating Label Variant", "Label Variant", label.Name)
	protoSource := label.Source
	source, err := serv.metaClient.GetSourceVariant(ctx, metadata.NameVariant{protoSource.Name, protoSource.Variant})
	if err != nil {
		return nil, err
	}
	label.Provider = source.Provider()
	return serv.meta.CreateLabelVariant(ctx, label)
}

func (serv *ApiServer) CreateTrainingSetVariant(ctx context.Context, train *pb.TrainingSetVariant) (*pb.Empty, error) {
	serv.Logger.Infow("Creating Training Set Variant", "Training Set Variant", train.Name)
	protoLabel := train.Label
	label, err := serv.metaClient.GetLabelVariant(ctx, metadata.NameVariant{protoLabel.Name, protoLabel.Variant})
	if err != nil {
		return nil, err
	}
	train.Provider = label.Provider()
	return serv.meta.CreateTrainingSetVariant(ctx, train)
}

func (serv *ApiServer) Serve() error {
	if serv.grpcServer != nil {
		return fmt.Errorf("Server already running")
	}
	lis, err := net.Listen("tcp", serv.address)
	if err != nil {
		return err
	}
	opts := []grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	}
	conn, err := grpc.Dial(serv.metaAddr, opts...)
	if err != nil {
		return err
	}
	serv.meta = pb.NewMetadataClient(conn)
	client, err := metadata.NewClient(serv.metaAddr, serv.Logger)
	if err != nil {
		return err
	}
	serv.metaClient = client
	return serv.ServeOnListener(lis)
}

func (serv *ApiServer) ServeOnListener(lis net.Listener) error {
	serv.listener = lis
	grpcServer := grpc.NewServer()
	pb.RegisterApiServer(grpcServer, serv)
	serv.grpcServer = grpcServer
	serv.Logger.Infow("Server starting", "Address", serv.listener.Addr().String())
	return grpcServer.Serve(lis)
}

func (serv *ApiServer) GracefulStop() error {
	if serv.grpcServer == nil {
		return fmt.Errorf("Server not running")
	}
	serv.grpcServer.GracefulStop()
	serv.grpcServer = nil
	serv.listener = nil
	return nil
}

func handleHealthCheck(w http.ResponseWriter, r *http.Request) {
	w.Header().Add("Strict-Transport-Security", "max-age=63072000; includeSubDomains")
	w.WriteHeader(http.StatusOK)

	_, err := io.WriteString(w, "OK")
	if err != nil {
		fmt.Printf("health check write response error: %+v", err)
	}

}

func handleIndex(w http.ResponseWriter, r *http.Request) {
	w.Header().Add("Strict-Transport-Security", "max-age=63072000; includeSubDomains")
	w.Header().Set("Content-Type", "text/html")
	w.WriteHeader(http.StatusOK)

	_, err := io.WriteString(w, `<html><body>Welcome to gRPC on GKE example</body></html>`)
	if err != nil {
		fmt.Printf("index / write response error: %+v", err)
	}

}

func startHttpsServer(port string) error {
	mux := &http.ServeMux{}

	// Health check endpoint will handle all /_ah/* requests
	// e.g. /_ah/live, /_ah/ready and /_ah/lb
	// Create separate routes for specific health requests as needed.
	mux.HandleFunc("/_ah/", handleHealthCheck)
	mux.HandleFunc("/", handleIndex)
	// Add more routes as needed.

	// Set timeouts so that a slow or malicious client doesn't hold resources forever.
	httpsSrv := &http.Server{
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 5 * time.Second,
		IdleTimeout:  60 * time.Second,
		Handler:      mux,
		Addr:         port,
	}

	fmt.Printf("starting HTTP server on port %s", port)

	return httpsSrv.ListenAndServe()
}

func main() {
	logger := zap.NewExample().Sugar()
	go func() {
		err := startHttpsServer(":8443")
		if err != nil && err != http.ErrServerClosed {
			panic(fmt.Sprintf("health check HTTP server failed: %+v", err))
		}
	}()
	serv, err := NewApiServer(logger, "0.0.0.0:7878", "sandbox-metadata-server:8080")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(serv.Serve())
}
