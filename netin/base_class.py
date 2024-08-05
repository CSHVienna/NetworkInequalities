from typing import Dict, Any, Optional
from datetime import datetime

class BaseClass:
    def __init__(self) -> None:
        self._created_at = datetime.now()

    def get_metadata(self, d_meta_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            self.__class__.__name__: {
                'created_at': self._created_at,
            }
        } if d_meta_data is None else d_meta_data
