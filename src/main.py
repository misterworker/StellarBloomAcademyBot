from collections import deque
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command

from build_graph import graph_builder
from config import CORS_ORIGINS
from helper import pool, create_prompt, AddAiMsgInput, UserInput, ValidateIdentityInput, WipeInput
from independents import validate_identity

import asyncio, json, os, sys

load_dotenv()
DB_URI = os.getenv("DB_URI")

# Fix for Windows event loop compatibility with psycopg async
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

graph = None
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Keep the connection pool open as long as the app is alive."""
    global graph
  
    checkpointer = AsyncPostgresSaver(pool)
    # await checkpointer.setup()
    graph = graph_builder.compile(checkpointer=checkpointer)
    
    # try:
    #     with open("./graph_output.png", "wb") as f:
    #         f.write(graph.get_graph().draw_mermaid_png())
    # except Exception as e:
    #     print("Exception while generating graph_output.png:", e)

    print("✅ Connection pool and graph initialized!")
    yield  # Yield control back to FastAPI while keeping the pool open

    await pool.close()
    print("❌ Connection pool closed!")

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def stream_graph_updates(user_id: str, user_input: str, config: dict, name: str, bot_name: str):
    state = {
        "messages": [SystemMessage(content=create_prompt(info=[name, bot_name], llm_type="chatbot")), {"role": "user", "content": user_input}],
        "user_id": user_id,
    }
    
    async for event in graph.astream(state, config):
        print("Event: ", event)
        msg = ""
        if "__interrupt__" in event:
            return {"response":"", "other_name":"interrupt", "other_msg": None} #TODO: Turn other msg into tool name

        elif "tools" in event:
            # For tools that don't require human review
            pass

        elif "splitter" in event:
            for value in event.values():
                msg = value["messages"][-1].content

        if msg:
            return {"response": msg, "other_name": None, "other_msg": None}

        
async def clear_thread(thread_id: str):
    """Deletes all records related to the given thread_id."""
    async with pool.connection() as conn:
        async with conn.cursor() as cursor:
            try:
                await cursor.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
                await cursor.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
                await cursor.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,))

                await conn.commit()
                # print(f"✅ Wiped data for thread_id: {thread_id}")
                return {"response": True, "other_name": None, "other_msg": None}

            except Exception as exception:
                await conn.rollback()
                print(f"❌ Error in wipe(): {exception}")
                return {"response": False, "other_name": None, "other_msg": None}


@app.post("/chat")
async def chat(input: UserInput):
    try:
        user_id = input.user_id
        user_input = input.user_input
        name = input.name
        bot_name = input.bot_name
        
        if not user_id:
            raise HTTPException(status_code=400, detail=f"Input not provided: {input}")

        config = {"configurable": {"thread_id": user_id}}
        
        result = await stream_graph_updates(user_id, user_input, config, name, bot_name)

        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/wipe")
# Should be used to wipe history when user leaves site or wants to
async def wipe(input: WipeInput):
    try:
        user_id = input.user_id
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID is required.")
        result = await clear_thread(user_id)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/identify")
async def identify(input: ValidateIdentityInput):
    try:
        is_valid = await validate_identity(input.user_input)
        return is_valid
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/add_ai_msg")
async def add_ai_msg(input: AddAiMsgInput):
    try:
        ai_msg = input.ai_msg
        user_id = input.user_id
        transformed_ai_msg = AIMessage(content=ai_msg)
        config = {"configurable": {"thread_id": user_id}}
        await graph.aupdate_state(config, {"messages": [transformed_ai_msg]})

        return {"response": "Successfully added ai message"}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
