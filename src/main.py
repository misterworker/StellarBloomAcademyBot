from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator

from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command

from build_graph import graph_builder
from config import CORS_ORIGINS, DISABLE_REWIND
from helper import create_prompt, pool, ResumeInput, UserInput, WipeInput

import os, sys, asyncio, json

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
    #     with open("../graph_output.png", "wb") as f:
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

async def stream_graph_updates(
    fingerprint: str,
    user_id: str,
    user_input: str,
    num_rewind: int,
    config: dict
) -> AsyncGenerator[str, None]:
    state = {
        "messages": [
            SystemMessage(content=create_prompt(info=[], llm_type="chatbot")),
            {"role": "user", "content": user_input}
        ],
        "user_id": user_id,
        "fingerprint": fingerprint,
    }

    if not DISABLE_REWIND:
        if num_rewind != 0:
            rewind(int(num_rewind), config, user_input)

    try:
        node = None; is_interrupt = False
        async for message, event in graph.astream(state, config, stream_mode="messages"):
            if not node:
                kwargs = message.additional_kwargs
                if not kwargs: node = "chat"
                elif kwargs.get("tool_calls", False):
                    node = "chat"
                    last_tool_call = message.tool_call_chunks[-1] if message.tool_call_chunks else {}
                    tool_name = last_tool_call.get("name", ValueError("No Tool Name"))
                    if tool_name != "suspend_user":
                        node = "int"
                        is_interrupt = True
                else: raise ValueError("additional_kwargs not found")

            if node == "chat": yield f"data: {json.dumps({'response': message.content})}\n\n"

        if is_interrupt:
            #! Interrupt AFTER stream finishes cleanly
            yield f"data: {json.dumps({'other_name': 'interrupt', 'other_msg': None})}\n\n"
            return
    except Exception as e:
        print(f"❌ Error in resume(): {e}")

async def resume_graph_updates(action, config):
    node = None
    try:
        async for message, resume_event in graph.astream(Command(resume={"action": action}), config, stream_mode="messages"):
            if not node: node = resume_event.get("langgraph_node", None)
            if node is None: raise ValueError(f"Missing langgraph_node in resume_event: {resume_event}")

            match node:
                case "chatbot": yield f"data: {json.dumps({'response': message.content, 'other_name': 'chat', 'other_msg': None})}\n\n"
                case "rag": yield f"data: {json.dumps({'response': message.content, 'other_name': 'rag', 'other_msg': None})}\n\n"
                case "email": yield f"data: {json.dumps({'response': message.content, 'other_name': 'email', 'other_msg': None})}\n\n"
    
    except Exception as e:
        print(f"❌ Error in resume(): {e}")
        
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
                new_message = HumanMessage(
                    content=user_input,
                    id=last_message.id
                )
                graph.update_state(config, {"messages": [new_message]})
                break
    # return config

@app.post("/chat")
async def chat(input: UserInput):
    try:
        if not input.user_id or not input.fingerprint:
            raise HTTPException(status_code=400, detail=f"Input not provided: {input}")

        config = {"configurable": {"thread_id": input.user_id}}

        return StreamingResponse(
            stream_graph_updates(
                input.fingerprint,
                input.user_id,
                input.user_input,
                input.num_rewind,
                config,
            ),
            media_type="text/event-stream"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/resume")
async def resume_process(input: ResumeInput):
    try:
        action = input.action
        user_id = input.user_id
        if action is None or not user_id:
            raise HTTPException(status_code=400, detail=f"Input not provided: {input}")
        config = {"configurable": {"thread_id": user_id}}
        return StreamingResponse(
            resume_graph_updates(
                action,
                config
            ),
            media_type="text/event-stream"
        )

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
    
