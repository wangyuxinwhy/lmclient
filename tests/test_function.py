from typing import Literal

from lmclient.function import function


def get_weather(city: str, country: Literal['US', 'CN'] = 'US') -> str:
    """
    Returns a string describing the weather in a given city and country.

    Args:
        city (str): The name of the city.
        country (str, optional): The name of the country. Defaults to 'US'.

    Returns:
        str: A string describing the weather in the given city and country.
    """
    return f'The weather in {city}, {country} is 72 degrees and sunny'


def test_function_decrator():
    wrapped = function(get_weather)
    json_schema = {
        'name': 'get_weather',
        'description': 'Returns a string describing the weather in a given city and country.',
        'parameters': {
            'properties': {
                'city': {'type': 'string', 'description': 'The name of the city.'},
                'country': {
                    'default': 'US',
                    'enum': ['US', 'CN'],
                    'type': 'string',
                    'description': "The name of the country. Defaults to 'US'.",
                },
            },
            'required': ['city'],
            'type': 'object',
        },
    }
    assert wrapped.json_schema == json_schema
