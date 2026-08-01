"""Inert stand-in for ``netplay.logging.video_renderer.AgentVideoRenderer``."""


class AgentVideoRenderer:
    def __init__(self, video_path=None, render=False):
        self.video_path = video_path
        self.render = render

    def init(self, observation=None):
        pass

    def add_step(self, observation=None, action=None, thoughts=None, is_ai_thought=False):
        pass

    def add_thoughts(self, thoughts=None, is_ai_thought=False):
        pass

    def close(self):
        pass
