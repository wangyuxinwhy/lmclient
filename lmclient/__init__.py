from lmclient.client import LMClient as LMClient
from lmclient.client import LMClientForStructuredData as LMClientForStructuredData
from lmclient.models import AzureChat as AzureChat
from lmclient.models import MinimaxChat as MinimaxChat
from lmclient.models import OpenAIChat as OpenAIChat
from lmclient.models import ZhiPuChat as ZhiPuChat
from lmclient.parsers import *  # noqa: F403

AzureCompletion = AzureChat
OpenAICompletion = OpenAIChat
