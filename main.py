from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from psycopg_pool import AsyncConnectionPool

from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command

from build_graph import graph_builder
from helper import create_prompt

import os

load_dotenv()
DB_URI = os.getenv("DB_URI_LOCAL")

graph = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Keep the connection pool open as long as the app is alive."""
    global graph
    pool = AsyncConnectionPool(
        conninfo=DB_URI,
        max_size=5,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    )
    checkpointer = AsyncPostgresSaver(pool)
    # await checkpointer.setup()
    graph = graph_builder.compile(checkpointer=checkpointer)
    
    # try:
    #     with open("graph_output.png", "wb") as f:
    #         f.write(graph.get_graph().draw_mermaid_png())
    # except Exception:
    #     # This requires some extra dependencies and is optional
    #     pass

    print("✅ Connection pool and graph initialized!")
    yield  # Yield control back to FastAPI while keeping the pool open

    await pool.close()
    print("❌ Connection pool closed!")

app = FastAPI(lifespan=lifespan)

class UserInput(BaseModel):
    user_input: str
    num_rewind: int = 0
    fingerprint: str

class ResumeInput(BaseModel):
    action: bool
    fingerprint: str

async def stream_graph_updates(user_input: str, fingerprint: str, num_rewind: int, config: dict):
    state = {
        "messages": [SystemMessage(content=create_prompt([230, 3], "chatbot")), {"role": "user", "content": user_input}],
        "fingerprint": fingerprint,
    }
    if num_rewind != 0:
        rewind(int(num_rewind), config, user_input)
    async for event in graph.astream(state, config):
        msg = ""
        if "__interrupt__" in event:
            return {"response":"", "other":"interrupt"}

        elif "tools" in event:
            # For tools that don't require human review
            pass

        elif "chatbot" in event:
            for value in event.values():
                msg = value["messages"][-1].content

        if msg:
            return {"response": msg, "other": None}

async def resume_graph_updates(action, config):
    msg = ""
    async for resume_event in graph.astream(Command(resume={"action": action}), config):
        try:
            is_chatbot = resume_event.get("chatbot", False)
            is_rag = resume_event.get("rag", False)
        except:
            is_chatbot = False
            is_rag = False
        if is_chatbot:
            msg = resume_event["chatbot"]["messages"][-1].content
        elif is_rag:
            msg = resume_event["rag"]["messages"][-1]
            
        if msg:
            return {"response": msg, "other": None}
        
        
def rewind(num_rewind:int, config, user_input):
    #TODO: fix. essentially need to clear old states instead of appending, which is what update_state seems to do, and also figure out the most effective way to time travel without relying on node type perhaps.
    #? Alternatively, this could also be ignored and we can provide our own logic of checkpoint ids and just use these ids.
    num_encountered = 0
    for state in graph.get_state_history(config):
        if "chatbot" in state.next:
            num_encountered+=1
            if num_rewind == num_encountered:
                config = state.config
                last_message = state.values["messages"][-1]
                print("last message: ", last_message)
                new_message = HumanMessage(
                    content=user_input,
                    id=last_message.id
                )
                graph.update_state(config, {"messages": [new_message]})
                break
    # return config
    
@app.post("/resume")
async def resume_process(input: ResumeInput):
    try:
        action = input.action
        fingerprint = input.fingerprint
        if action is None:
            raise HTTPException(status_code=400, detail="Action is required.")
        if not fingerprint:
            raise HTTPException(status_code=400, detail="Fingerprint is required.")
        config = {"configurable": {"thread_id": fingerprint}}
        result = await resume_graph_updates(action, config)
        print("resume result: ", result)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(input: UserInput):
    try:
        user_input = input.user_input
        num_rewind = input.num_rewind
        fingerprint = input.fingerprint
        if not fingerprint:
            raise HTTPException(status_code=400, detail="Fingerprint is required.")
        config = {"configurable": {"thread_id": fingerprint}}
        
        result = await stream_graph_updates(user_input, fingerprint, num_rewind, config)
        print("chat result: ", result)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
