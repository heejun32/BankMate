class CurrentUser:
    def __init__(self):
        self.current_user = 0

    def set_current_user(self, username):
        self.current_user = username

    def init_current_user(self):
        self.current_user = 0

    def who_is_now(self):
        return self.current_user