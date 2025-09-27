from datetime import datetime
from typing import Dict, Any


class AppState:
    """Class for managing application state"""
    
    def __init__(
        self,
        user_name: str = "Николай",
        persona: str = "business",
        default_city: str = "Bad Mergentheim"
    ):
        self.user_name = user_name
        self.persona = persona
        self.default_city = default_city
        self._update_datetime_info()
    
    def _update_datetime_info(self) -> None:
        """Updates current date and time information"""
        now = datetime.now()
        self.current_date = now.strftime("%d.%m.%Y")
        self.current_time = now.strftime("%H:%M")
        self.current_weekday = now.strftime("%A")
    
    def get_state(self) -> Dict[str, Any]:
        """Returns dictionary with current application state"""
        # Update time information on each call
        self._update_datetime_info()
        
        return {
            "user_name": self.user_name,
            "persona": self.persona,
            "default_city": self.default_city,
            "current_date": self.current_date,
            "current_time": self.current_time,
            "current_weekday": self.current_weekday,
        }


def get_state(user_name: str = "Николай", persona: str = "business") -> Dict[str, Any]:
    """Function for getting application state"""
    app_state = AppState(user_name=user_name, persona=persona)
    return app_state.get_state()
