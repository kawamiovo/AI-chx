from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import re

from config import Config
from main_agent import MainAgent
from llm import LLMService
from conversation_history import ConversationHistory
import asyncio
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# 初始化 LLM 服务和 Agent
llm_service = LLMService(
    api_key=Config.LLM_API_KEY,
    api_url=Config.LLM_API_URL
)
agent = MainAgent(llm_service, ConversationHistory())

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/get_chat_history')
def get_chat_history():
    # 调用刚才写的方法获取最近对话列表
    history = agent.conversation_history.get_history_list()
    return jsonify(history)
@socketio.on('send_message')
def handle_message(data):
    message = data['message']
    
    # 创建新的事件循环来运行异步函数
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # 获取回复
        reply, expression = loop.run_until_complete(agent.reply(message))

        # 如果reply已经是字典,直接使用
        if isinstance(reply, dict):
            actual_reply = reply.get('reply', reply)
        else:
            # 尝试解析JSON
            try:
                # 使用ast.literal_eval更安全地解析
                import ast
                reply_dict = ast.literal_eval(reply)
                actual_reply = reply_dict.get('reply', reply)
            except:
                try:
                    reply_json = json.loads(reply)
                    actual_reply = reply_json.get('reply', reply)
                except json.JSONDecodeError:
                    actual_reply = reply

        emit('receive_message', {
            'message': actual_reply,
            'expression': expression,
            'isBot': True
        })
    except Exception as e:
        print(f"Error handling message: {str(e)}")
        emit('receive_message', {
            'message': "对不起，我遇到了一些问题，请稍后再试。",
            'expression': "生气",
            'isBot': True
        })
    finally:
        # 关闭事件循环
        loop.close()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True,allow_unsafe_werkzeug=True)
