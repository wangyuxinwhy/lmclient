from typing import Literal

from pydantic import BaseModel

from lmclient.function import function, get_json_schema


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


class UserInfo(BaseModel):
    name: str
    age: int


@function
def upload_user_info(user_info: UserInfo) -> Literal['success']:
    """
    Uploads user information to the database.

    Args:
        user_info (UserInfo): The user information to be uploaded.
    """
    return 'success'


def test_get_json_schema() -> None:
    json_schema = get_json_schema(get_weather)
    expected_json_schema = {
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
    assert json_schema == expected_json_schema


def test_validate_function() -> None:
    output = upload_user_info(user_info={'name': 'John', 'age': 20})  # type: ignore
    assert output == 'success'
