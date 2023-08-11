from lmclient.client import LMClient as LMClient
from lmclient.client import LMClientForStructuredData as LMClientForStructuredData
from lmclient.models import AzureChat as AzureChat
from lmclient.models import MinimaxChat as MinimaxChat
from lmclient.models import OpenAIChat as OpenAIChat
from lmclient.parsers import *
from lmclient.version import __cache_version__, __version__

AzureCompletion = AzureChat
OpenAICompletion = OpenAIChat
