from nanobot.channels.base import Channel
from nanobot.agent.loop import AgentLoop

class EdgeChannel(Channel):
    name = "edge"

    def __init__(self, agent_loop: AgentLoop):
        super().__init__(agent_loop)

    def handle_message(self, text: str) -> str:
        """
        Synchronous request/response for edge devices
        """
        return self.agent_loop.run(text)
