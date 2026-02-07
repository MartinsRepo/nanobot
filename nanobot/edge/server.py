from flask import Flask, request, jsonify
from nanobot.agent.loop import AgentLoop
from nanobot.channels.edge import EdgeChannel

app = Flask(__name__)

agent_loop = AgentLoop()
channel = EdgeChannel(agent_loop)

@app.route("/prompt", methods=["POST"])
def prompt():
    data = request.get_json(force=True)
    text = data.get("text", "")
    reply = channel.handle_message(text)
    return jsonify({"response": reply})

def main():
    app.run(host="127.0.0.1", port=5000)

if __name__ == "__main__":
    main()
