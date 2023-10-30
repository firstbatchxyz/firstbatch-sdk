"""Constants for the firstbatch package."""

EU_CENTRAL_1 = "https://aws-eu-central-1.hollowdb.xyz/"
US_WEST_1 = "https://aws-us-west-1.hollowdb.xyz/"
US_EAST_1 = "https://aws-us-east-1.hollowdb.xyz/"
ASIA_PACIFIC_1 = "https://aws-ap-southeast-1.hollowdb.xyz/"

regions = {"us-east-1":US_EAST_1, "us-west-1":US_WEST_1, "eu-central-1":EU_CENTRAL_1, "ap-southeast-1": ASIA_PACIFIC_1}

REGION_URL = "https://idp.firstbatch.xyz/v1/teams/team/get-team-information"

DEFAULT_QUANTIZER_TRAIN_SIZE = 100
DEFAULT_QUANTIZER_TYPE = "scalar"
DEFAULT_EMBEDDING_SIZE = 1536
DEFAULT_CONFIDENCE_INTERVAL_RATIO = 0.15
DEFAULT_COLLECTION = "my_collection"
DEFAULT_BATCH_SIZE = 10
DEFAULT_KEY = "text"
DEFAULT_TOPK_QUANT = 5
MINIMUM_TOPK = 5
DEFAULT_HISTORY = False
DEFAULT_VERBOSE = False
DEFAULT_HISTORY_FIELD = "id"

