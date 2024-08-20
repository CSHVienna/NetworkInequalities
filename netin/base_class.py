from typing import Dict, Any, Optional
from datetime import datetime

class BaseClass:
    _verbose: bool

    def __init__(self, verbose: bool = False) -> None:
        self._created_at = datetime.now()
        self._verbose = verbose

    def get_metadata(self, d_meta_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            self.__class__.__name__: {
                'created_at': self._created_at,
            }
        } if d_meta_data is None else d_meta_data

    def log(self, msg: str):
        if self._verbose:
            print(msg)
