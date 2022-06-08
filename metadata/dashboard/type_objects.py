from ssl import create_default_context


class FeatureVariantResource:
    def __init__(self, 
        created=None, 
        description="", 
        entity="", 
        name="", 
        owner="", 
        provider="", 
        dataType="", 
        variant="",
        status="",
        location=None,
        source=None,
        trainingSets=None):

        self.created = created
        self.description = description
        self.entity = entity
        self.name = name
        self.owner = owner
        self.provider = provider
        self.dataType = dataType
        self.variant = variant
        self.status = status
        self.location = location
        self.source = source
        self.trainingSets = trainingSets

class FeatureResource:
    def __init__(self, 
        name="",
        defaultVariant="",
        type = "",
        variants=None,
        allVariants=[]):

        self.allVariants = allVariants
        self.type = type
        self.defaultVariant = defaultVariant
        self.name = name
        self.variants = variants

class TrainingSetVariantResource:
    def __init__(self, 
        created=None, 
        description="", 
        name="", 
        owner="", 
        provider="", 
        variant="",
        label=None,
        features=None,
        status=""):

        self.created = created
        self.description = description
        self.name = name
        self.owner = owner
        self.provider = provider
        self.variant = variant
        self.status = status
        self.label = label
        self.features = features

class TrainingSetResource:
    def __init__(self, 
        type = "",
        defaultVariant="",
        name="",
        variants=None,
        allVariants=[]):
        self.allVariants = allVariants
        self.type = type
        self.defaultVariant = defaultVariant
        self.name = name
        self.variants = variants

class SourceVariantResource:
    def __init__(self, 
        created=None, 
        description="", 
        name="", 
        sourceType = "",
        owner="", 
        provider="", 
        variant="",
        status="",
        definition="",
        labels=None,
        features=None,
        trainingSets=None):

        self.created = created
        self.description = description
        self.name = name
        self.sourceType = sourceType
        self.owner = owner
        self.provider = provider
        self.variant = variant
        self.status = status
        self.definition = definition
        self.labels = labels
        self.features = features
        self.trainingSets = trainingSets

class SourceResource:
    def __init__(self, 
        type = "",
        defaultVariant="",
        name="",
        variants=None,
        allVariants=[]):
        self.allVariants = allVariants
        self.type = type
        self.defaultVariant = defaultVariant
        self.name = name
        self.variants = variants

class LabelVariantResource:
    def __init__(self, 
        created=None, 
        description="",
        entity="",
        name="", 
        owner="", 
        provider="",
        dataType = "", 
        variant="",
        location=None,
        source=None,
        status="",
        trainingSets=None):
        

        self.created = created
        self.description = description
        self.entity = entity
        self.dataType = dataType
        self.name = name
        self.owner = owner
        self.provider = provider
        self.variant = variant
        self.status = status
        self.source = source
        self.location = location
        self.trainingSets = trainingSets

class LabelResource:
    def __init__(self, 
        type = "",
        defaultVariant="",
        name="",
        variants=None,
        allVariants=[]):
        self.allVariants = allVariants
        self.type = type
        self.defaultVariant = defaultVariant
        self.name = name
        self.variants = variants

class EntityResource:
    def __init__(self, 
        description="",
        type="",
        name="", 
        features=None,
        labels=None,
        trainingSets=None,
        status=""):

        self.description = description
        self.type = type
        self.name = name
        self.features = features
        self.labels = labels
        self.trainingSets = trainingSets
        self.status = status

class UserResource:
    def __init__(self, 
        name="",
        type="",
        features=None,
        labels=None,
        trainingSets=None,
        sources=None,
        status=""):

        self.name = name
        self.type = type
        self.features = features
        self.labels = labels
        self.trainingSets = trainingSets
        self.sources = sources
        self.status = status 

class ModelResource:
    def __init__(self, 
        name="",
        type="",
        description="",
        features=None,
        labels=None,
        trainingSets=None,
        status=""):

        self.name = name
        self.type = type
        self.description = description
        self.features = features
        self.labels = labels
        self.trainingSets = trainingSets
        self.status = status 

class ProviderResource:
    def __init__(self, 
        name="",
        type="",
        description="",
        providerType="",
        software="",
        team="",
        sources=None,
        features=None,
        labels=None,
        trainingSets=None,
        status="",
        serializedConfig=""):

        self.name = name
        self.type = type
        self.description = description
        self.providerType=providerType
        self.software=software
        self.team=team
        self.sources=sources
        self.features = features
        self.labels = labels
        self.trainingSets = trainingSets
        self.status = status 
        self.serializedConfig=serializedConfig