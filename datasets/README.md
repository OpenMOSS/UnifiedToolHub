# Datasets 和 Tools

## 数据集与调用工具

收集相关的数据集及其对应的调用工具文件。当前已整理的数据集包括：

**BFCL**


## 标准格式示例【对于BFCL版本】：
```json
{
    "id": "live_simple_136-89-0",
    "query":  "I need to retrieve the topology information of the SalesApp under the AcmeCorp account. Could you send a GET request to the server at 'https://192.168.1.1/api/v1/applications/topologies' using the filter 'accountName:AcmeCorp AND applicationName:SalesApp'?",
    "candidate_apis": [
        {
            "name": "requests.get",
            "description": "Sends a GET request to the specified URL to retrieve the topology information of an application.",
            "parameters": {
                "type": "dict",
                "required": [
                    "url",
                    "params"
                ],
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL endpoint from which to retrieve the application topology. The URL must be a valid HTTP or HTTPS URL, formatted as 'https://{ip}/api/v1/applications/topologies' where {ip} is the IP address of the server."
                    },
                    "params": {
                        "type": "dict",
                        "properties": {
                            "filter": {
                                "type": "string",
                                "description": "A filter in Lucene format, specifying the accountName, controllerName, applicationName. in the format of accountName:DefaultCorp AND applicationName:DefaultApp"
                            }
                        },
                        "description": "Query parameters to include in the GET request. The 'filter' field specifies search criteria and is optional with a default value if not provided."
                    }
                }
            }
        }
    ],
    "ground_truth": [
        {
            "requests.get": {
                "url": [
                    "https://192.168.1.1/api/v1/applications/topologies"
                ],
                "params": [
                    {
                        "filter": [
                            "accountName:AcmeCorp AND applicationName:SalesApp"
                        ]
                    }
                ]
            }
        }
    ]
}
```

## MTU-Bench标准【多轮对话】
```
"ground_truth": [
    {
        "query": [
            {
                "User": "Hey, could you help me with some temperature settings in my house?",
                "Assistant": "Of course, how can I assist you with your temperature settings?"
            },
            {
                "User": "Can you tell me the current temperature in a particular room?",
                "Assistant": "Sure, which room are you referring to?"
            },
            {
                "User": "The living room. Also, can you set the temperature of the bedroom to match that of the living room?"
            }
        ],
        "candidate_apis": [
            {
                "type": "function",
                "function": {
                    "name": "get_current_temperature",
                    "description": "Retrieves the current ambient temperature from a specified location, such as a room, building, or outdoor environment.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The specified location, such as a room, building, or outdoor environment from where the current ambient temperature is to be retrieved."
                            }
                        },
                        "required": [
                            "location"
                        ]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_door_status",
                    "description": "Queries the status of a door to determine if it is currently open or closed, and may provide additional security information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "door_location": {
                                "type": "string",
                                "description": "The location of the door that the user wants to check the status of"
                            }
                        },
                        "required": [
                            "door_location"
                        ]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "play_music",
                    "description": "Initiates the playback of music on a connected audio system, including options to select playlists, songs, or streaming services.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "song/playlist": {
                                "type": "string",
                                "description": "The specific song or playlist that the user wants to play."
                            },
                            "volume": {
                                "type": "float",
                                "description": "The volume at which the music should be played."
                            },
                            "music_service": {
                                "type": "string",
                                "description": "The specific music or streaming service, e.g. Spotify."
                            },
                            "location": {
                                "type": "string",
                                "description": "The location where the user wants the music to be played."
                            },
                            "music_genre": {
                                "type": "string",
                                "description": "The specific genre of music the user wants to play."
                            }
                        },
                        "required": [
                            "song/playlist"
                        ]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "lock_door",
                    "description": "Commands a smart lock to secure a door, ensuring the door is locked and reporting the lock status.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "door_name": {
                                "type": "string",
                                "description": "The name or location of the door that needs to be locked"
                            }
                        },
                        "required": [
                            "door_name"
                        ]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "adjust_room_temperature",
                    "description": "Controls the thermostat or heating/cooling system to set a specific temperature for a room or area.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "temperature": {
                                "type": "integer",
                                "description": "The desired temperature to set the thermostat or heating/cooling system to."
                            },
                            "location": {
                                "type": "string",
                                "description": "The specific room or area where the temperature should be adjusted."
                            }
                        },
                        "required": [
                            "temperature",
                            "location"
                        ]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "adjust_lighting",
                    "description": "Modifies the lighting conditions in a specified area, which can include dimming, color changes, or turning lights on or off.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "Specifies the area where the lighting conditions are to be modified"
                            },
                            "action": {
                                "type": "string",
                                "description": "Determines the action to be executed on the specified lighting conditions such as turning on or off lights or dimming"
                            },
                            "intensity": {
                                "type": "float",
                                "description": "Represents the level of brightness to be set if the lights are to be dimmed or brightened"
                            }
                        },
                        "required": [
                            "location",
                            "action"
                        ]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "adjust_music_volume",
                    "description": "Changes the volume of the music currently being played on a connected audio system.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "volume": {
                                "type": "float",
                                "description": "A numerical representation of the desired music volume"
                            }
                        },
                        "required": [
                            "volume"
                        ]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "GetWeather",
                    "description": "Get the weather of a certain location on a date. The format for values of parameters related to date is \"^\\d{4}-\\d{2}-\\d{2}$\", and for parameters related to time, it is \"HH:MM\". Values for parameters indicating yes or no should use the boolean type.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
        ],
        "response": [
            {
                "name": "get_current_temperature",
                "arguments": {
                    "location": "living room"
                }
            },
            {
                "name": " adjust_room_temperature",
                "arguments": {
                    "temperature": "get_current_temperature.temperature",
                    "location": "bedroom"
                }
            }
        ]
    },
]
```
