FastAPI documentation: https://fastapi.tiangolo.com

This tutorial app source code (GitLab repo): Please, for more comprehension access the repo https://gitlab.com/rhkina/fastapi as you follow this tutorial!

FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints.

Historically, async work in Python has been nontrivial (though its API has rapidly improved since Python 3.4) particularly with Flask. Essentially, Flask (on most WSGI servers) is blocking by default - work triggered by a request to a particular endpoint will hold the server entirely until that request is completed. Instead, Flask (or rather, the WSGI server running it, like gunicorn or uWSGI) achieve scaling by running multiple worker instances of the app in parallel, such that requests can be farmed to other workers while one is busy. Within a single worker, asynchronous work can be wrapped in a blocking call (the route function itself is still blocking), threaded (in newer versions of Flask), or farmed to a queue manager like Celery - but there isn’t a single consistent story where routes can cleanly handle asynchronous requests without additional tooling.

FastAPI is designed from the ground up to run asynchronously - thanks to its underlying starlette ASGI framework, route functions default to running within an asynchronous event loop. With a good ASGI server (FastAPI is designed to couple to uvicorn, running on top of uvloop) this can get us performance on par with fast asynchronous webservers in Go or Node, without losing the benefits of Python’s broader machine learning ecosystem.

In contrast to messing with threads or Celery queues to achieve asynchronous execution in Flask, running an endpoint asynchronously is dead simple in FastAPI - we simply declare the route function as asynchronous (with async def) and we’re ready to go! We can even do this if the route function isn’t conventionally asynchronous - that is, we don’t have any awaitable calls (like if the endpoint is running inference against an ML model). In fact, unless the endpoint is specifically performing a blocking IO operation (to a database, for example), it’s better to declare the function with async def (as blocking functions are actually punted to an external threadpool and then awaited anyhow).

The key features are:

Fast: Very high performance, on par with NodeJS and Go (thanks to Starlette and Pydantic). One of the fastest Python frameworks available.

Fast to code: Increase the speed to develop features by about 200% to 300%.*

Fewer bugs: Reduce about 40% of human (developer) induced errors.*
Intuitive: Great editor support. Completion everywhere. Less time debugging.
Easy: Designed to be easy to use and learn. Less time reading docs.
Short: Minimize code duplication. Multiple features from each parameter declaration. Fewer bugs.
Robust: Get production-ready code. With automatic interactive documentation.
Standards-based: Based on (and fully compatible with) the open standards for APIs: OpenAPI (previously known as Swagger) and JSON Schema.
* estimation based on tests performed by internal development team.

Tutorial
Note: original blog to this tutorial from Michael Herman at testdriven.io.

Objectives
I have the following objectives with this repo:

Develop an asynchronous RESTful API with Python and FastAPI
Practice Test-Driven Development
Test a FastAPI app with Pytest
Interact with a Postgres database asynchronously
Containerize FastAPI and Postgres inside a Docker container
Parameterize test functions and mock functionality in tests with Pytest
Document a RESTful API with Swagger/OpenAPI
Project Setup
Create the following structure:

 fastapi
    ├── docker-compose.yml
    └── src
        ├── Dockerfile
        ├── app
        │   ├── __init__.py
        │   └── main.py
        └── requirements.txt
Add FastAPI and Uvicorn to the requirements file:

fastapi==0.54.1
uvicorn==0.11.3
Within main.py, create a new instance of FastAPI and set up a sanity check route:

from fastapi import FastAPI

app = FastAPI()

@app.get("/ping")
def pong():
    return {"ping": "pong!"}
Add the following lines in the Dockerfile (at src directory):

# pull official base image
FROM python:3.8.1-alpine

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# copy requirements file
COPY ./requirements.txt /usr/src/app/requirements.txt

# install dependencies
RUN set -eux \
    && apk add --no-cache --virtual .build-deps build-base \
        libressl-dev libffi-dev gcc musl-dev python3-dev \
    && pip install --upgrade pip setuptools wheel \
    && pip install -r /usr/src/app/requirements.txt \
    && rm -rf /root/.cache/pip

# copy project
COPY . /usr/src/app/
Here we started an Alpine-based Docker image for Python 3.8.1 and set a working directory defining two environment variables:

PYTHONDONTWRITEBYTECODE: Prevents Python from writing pyc files to disc (equivalent to python -B option)
PYTHONUNBUFFERED: Prevents Python from buffering stdout and stderr (equivalent to python -u option)
Finally, we copied over the requirements.txt file, installed some system-level dependencies, updated Pip, installed the requirements, and copied over the FastAPI app itself.

Next, add the following to the docker-compose.yml file in the project root:

version: '2.0'

services:
  web:
    build: ./src
    command: uvicorn app.main:app --reload --workers 1 --host 0.0.0.0 --port 8000
    volumes:
      - ./src/:/usr/src/app/
    ports:
      - 8002:8000
So, when the container spins up, Uvicorn will run with the following settings:

--reload enables auto-reload so the server will restart after changes are made to the code base.
--workers 1 provides a single worker process.
--host 0.0.0.0 defines the address to host the server on.
--port 8000 defines the port to host the server on.
app.main:app tells Uvicorn where it can find the FastAPI ASGI application - e.g., within the 'app' module, you'll find the ASGI app, app = FastAPI(), in the main.py file.

Build the image and spin up the container:

$ docker-compose up -d --build

Navigate to http://localhost:8002/ping. You should see: {"ping":"pong!"}

You'll also be able to view the interactive API documentation, powered by Swagger UI, at http://localhost:8002/docs:



Test setup
Create a tests folder in src and then add an __init__.py file to tests along with a test_main.py file:

from starlette.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_ping():
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"ping": "pong!"}
Here, we imported Starlette's TestClient, which uses the Requests library to make requests against the FastAPI app.

Add Pytest and Requests to requirements.txt:

fastapi==0.54.1
uvicorn==0.11.3

# dev
pytest==5.4.1
requests==2.23.0
Update the image and then run the tests:

$ docker-compose up -d --build
$ docker-compose exec web pytest .
You should see something like this:

========================================== test session starts ==========================================
platform linux -- Python 3.8.1, pytest-5.4.1, py-1.8.1, pluggy-0.13.1
rootdir: /usr/src/app
collected 1 item                                                                                        

tests/test_main.py .                                                                              [100%]

=========================================== 1 passed in 0.48s ===========================================
Add a test_app Pytest fixture to a new file called src/tests/conftest.py:

import pytest
from starlette.testclient import TestClient
from app.main import app

@pytest.fixture(scope="module")
def test_app():
    client = TestClient(app)
    yield client  # testing happens here
Update the test_main.py file as well so that it uses the fixture:

def test_ping(test_app):
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"ping": "pong!"}
Your project structure should now look like this:

fastapi
    ├── docker-compose.yml
    └── src
        ├── Dockerfile
        ├── app
        │   ├── __init__.py
        │   └── main.py
        ├── requirements.txt
        └── tests
            ├── __init__.py
            ├── conftest.py
            └── test_main.py
Async Handlers
Let's convert the synchronous handler over to an asynchronous one.

Rather than having to go through the trouble of spinning up a task queue (like Celery or RQ) or utilizing threads, FastAPI makes it easy to deliver routes asynchronously. As long as you don't have any blocking I/O calls in the handler, you can simply declare the handler as asynchronous by just adding the async keyword like so (modify main.py):

@app.get("/ping")
async def pong():
    # some async operation could happen here
    # example: `notes = await get_all_notes()`
    return {"ping": "pong!"}
That's it! Update the handler in your code, and then make sure the tests still pass:

$ docker-compose up -d --build
$ docker-compose exec web pytest .
========================================== test session starts ==========================================
platform linux -- Python 3.8.1, pytest-5.4.1, py-1.8.1, pluggy-0.13.1
rootdir: /usr/src/app
collected 1 item                                                                                        

tests/test_main.py .                                                                              [100%]

=========================================== 1 passed in 0.03s ===========================================
Routes
Next, let's set up the basic CRUD routes, following RESTful best practices:

Endpoint	HTTP Method	CRUD Method	Result
/notes/	GET	READ	get all notes
/notes/:id	GET	READ	get a single note
/notes/	POST	CREATE	add a note
/notes/:id	PUT	UPDATE	update a note
/notes/:id	DELETE	DELETE	delete a note

For each route, we will:

Write a test
Run the test to ensure it fails (red)
Write just enough code to get the test to pass (green)
Refactor
Before diving in, let's add some structure to better organize the CRUD routes with FastAPI's APIRouter.

You can break up and modularize larger projects as well as apply versioning to your API with the APIRouter. If you're familiar with Flask, it is equivalent to a Blueprint.

Add a new folder called api to the app folder, and add an __init__.py file to the newly created folder.

Now we can move the /ping route to a new file called src/app/api/ping.py:

from fastapi import APIRouter

router = APIRouter()

@router.get("/ping")
async def pong():
    # some async operation could happen here
    # example: `notes = await get_all_notes()`
    return {"ping": "pong!"}
Then, update main.py like so to remove the old route and wire the router up to our main app:

from fastapi import FastAPI
from app.api import ping

app = FastAPI()

app.include_router(ping.router)
Rename test_main.py to test_ping.py.

Make sure http://localhost:8002/ping and http://localhost:8002/docs still work. Also, be sure the tests still pass before moving on.

$ docker-compose up -d --build
$ docker-compose exec web pytest .
========================================== test session starts ==========================================
platform linux -- Python 3.8.1, pytest-5.4.1, py-1.8.1, pluggy-0.13.1
rootdir: /usr/src/app
collected 1 item                                                                                        

tests/test_ping.py .                                                                              [100%]

=========================================== 1 passed in 0.04s ===========================================
Postgres setup
To configure Postgres, we'll need to add a new service to the docker-compose.yml file, add the appropriate environment variables, and install asyncpg.

First, add a new service called db to docker-compose.yml:

version: '2.0'

services:
  web:
    build: ./src
    command: uvicorn app.main:app --reload --workers 1 --host 0.0.0.0 --port 8000
    volumes:
      - ./src/:/usr/src/app/
    ports:
      - 8002:8000
    environment:
      - DATABASE_URL=postgresql://hello_fastapi:hello_fastapi@db/hello_fastapi_dev
  db:
    image: postgres:12.1-alpine
    volumes:
      - postgres_data:/var/lib/postgresql/data/
    environment:
      - POSTGRES_USER=hello_fastapi
      - POSTGRES_PASSWORD=hello_fastapi
      - POSTGRES_DB=hello_fastapi_dev

volumes:
  postgres_data:
To persist the data beyond the life of the container we configured a volume. This config will bind postgres_data to the /var/lib/postgresql/data/ directory in the container.

We also added an environment key to define a name for the default database and set a username and password.

Review the "Environment Variables" section of the Postgres Docker Hub page for more info.

Update the Dockerfile to install the appropriate packages required for asyncpg:

# pull official base image
FROM python:3.8.1-alpine

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# copy requirements file
COPY ./requirements.txt /usr/src/app/requirements.txt

# install dependencies
RUN set -eux \
    && apk add --no-cache --virtual .build-deps build-base \
        libressl-dev libffi-dev gcc musl-dev python3-dev \
        postgresql-dev \
    && pip install --upgrade pip setuptools wheel \
    && pip install -r /usr/src/app/requirements.txt \
    && rm -rf /root/.cache/pip

# copy project
COPY . /usr/src/app/
Add asyncpg to src/requirements.txt:

asyncpg==0.20.0
fastapi==0.54.1
uvicorn==0.11.3

# dev
pytest==5.4.1
requests==2.23.0
Next, add a db.py file to src/app:

import os

from databases import Database
from sqlalchemy import create_engine, MetaData

DATABASE_URL = os.getenv("DATABASE_URL")

# SQLAlchemy
engine = create_engine(DATABASE_URL)
metadata = MetaData()

# databases query builder
database = Database(DATABASE_URL)
Here, using the database URI and credentials that we just configured in the Docker Compose file, we created a SQLAlchemy engine (used for communicating with the database) along with a Metadata instance (used for creating the database schema). We also created a new Database instance from databases.

databases is an async SQL query builder that works on top of the SQLAlchemy Core expression language. It supports the following methods:

database.fetch_all(query)
database.fetch_one(query)
database.iterate(query)
database.execute(query)
database.execute_many(query)
Review the Async SQL (Relational) Databases guide and the Starlette Database docs for more details on working with databases asynchronously.

Update the requirements.txt:

asyncpg==0.20.0
databases[postgresql]==0.2.6
fastapi==0.54.1
SQLAlchemy==1.3.16
uvicorn==0.11.3

# dev
pytest==5.4.1
requests==2.23.0
Models
SQLAlchemy Model
Add a notes model to src/app/db.py:

import os

from sqlalchemy import (Column, DateTime, Integer, MetaData, String, Table,
                        create_engine)
from sqlalchemy.sql import func
from databases import Database

DATABASE_URL = os.getenv("DATABASE_URL")

# SQLAlchemy
engine = create_engine(DATABASE_URL)
metadata = MetaData()
notes = Table(
    "notes",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("title", String(50)),
    Column("description", String(50)),
    Column("created_date", DateTime, default=func.now(), nullable=False),
)

# databases query builder
database = Database(DATABASE_URL)
Wire up the database and the model in main.py and add startup and shutdown event handlers for connecting to and disconnecting from the database:

from fastapi import FastAPI
from app.api import notes, ping
from app.db import engine, metadata, database

metadata.create_all(engine)

app = FastAPI()

@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

app.include_router(ping.router)
Build the new image and spin up the two containers:

$ docker-compose up -d --build

Ensure the notes table was created:

$ docker-compose exec db psql --username=hello_fastapi --dbname=hello_fastapi_dev

psql (12.1)
Type "help" for help.

hello_fastapi_dev=# \l
                                            List of databases
       Name        |     Owner     | Encoding |  Collate   |   Ctype    |        Access privileges        
-------------------+---------------+----------+------------+------------+---------------------------------
 hello_fastapi_dev | hello_fastapi | UTF8     | en_US.utf8 | en_US.utf8 | 
 postgres          | hello_fastapi | UTF8     | en_US.utf8 | en_US.utf8 | 
 template0         | hello_fastapi | UTF8     | en_US.utf8 | en_US.utf8 | =c/hello_fastapi               +
                   |               |          |            |            | hello_fastapi=CTc/hello_fastapi
 template1         | hello_fastapi | UTF8     | en_US.utf8 | en_US.utf8 | =c/hello_fastapi               +
                   |               |          |            |            | hello_fastapi=CTc/hello_fastapi
(4 rows)

hello_fastapi_dev=# \c hello_fastapi_dev
You are now connected to database "hello_fastapi_dev" as user "hello_fastapi".
hello_fastapi_dev=# \dt
           List of relations
 Schema | Name  | Type  |     Owner     
--------+-------+-------+---------------
 public | notes | table | hello_fastapi
(1 row)

hello_fastapi_dev=# \q
Pydantic model
First time using pydantic? View pydantic Overview for more details.

Create a NoteSchema pydantic model with two required fields, title and description, in a new file called models.py in src/app/api:

from pydantic import BaseModel

class NoteSchema(BaseModel):
    title: str
    description: str

NoteSchema will be used for validating the payloads for creating and updating notes.

POST route
We will break the normal TDD flow for this route in order to establish the coding pattern that we'll use for the remaining routes.

Code
Create a new file called notes.py in the src/app/api folder:

from app.api import crud
from app.api.models import NoteDB, NoteSchema
from fastapi import APIRouter, HTTPException

router = APIRouter()

@router.post("/", response_model=NoteDB, status_code=201)
async def create_note(payload: NoteSchema):
    note_id = await crud.post(payload)
    response_object = {
        "id": note_id,
        "title": payload.title,
        "description": payload.description,
    }
    return response_object
Here, we defined a handler that expects a payload, payload: NoteSchema, with a title and a description.

Essentially, when the route is hit with a POST request, FastAPI will read the body of the request and validate the data: - If valid, the data will be available in the payload parameter. FastAPI also generates JSON Schema definitions that are then used to automatically generate the OpenAPI schema and the API documentation. - If invalid, an error is immediately returned.

Review the Request Body for more info.

Note that we used the async declaration here since the database communication will be asynchronous. In other words, there are no blocking I/O operations in the handler.

Next, create a new file called crud.py in the src/app/api folder:

from app.api.models import NoteSchema
from app.db import notes, database

async def post(payload: NoteSchema):
    query = notes.insert().values(title=payload.title, description=payload.description)
    return await database.execute(query=query)
We added a utility function called post for creating new notes that takes a payload object and then:

Creates a SQLAlchemy insert object expression query.
Executes the query and returns the generated ID.
Next, we need to define a new pydantic model for use as the response_model:

@router.post("/", response_model=NoteDB, status_code=201)

Update models.py like so:

from pydantic import BaseModel

class NoteSchema(BaseModel):
    title: str
    description: str

class NoteDB(NoteSchema):
    id: int
The NoteDB model inherits from the NoteSchema model, adding an id field.

Wire up the new router in main.py:

from app.api import notes, ping
from app.db import database, engine, metadata
from fastapi import FastAPI

metadata.create_all(engine)

app = FastAPI()

@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

app.include_router(ping.router)
app.include_router(notes.router, prefix="/notes", tags=["notes"])
Take note of the prefix URL along with the notes tag, which will be applied to the OpenAPI schema (for grouping operations).

Test it out with curl or HTTPie: $ http --json POST http://localhost:8002/notes/ title=foo description=bar

You should see something like:

HTTP/1.1 201 Created
content-length: 42
content-type: application/json
date: Sun, 12 Apr 2020 04:39:56 GMT
server: uvicorn

{
    "description": "bar",
    "id": 1,
    "title": "foo"
}
You can also interact with the endpoint at http://localhost:8002/docs/.

FastAPI

Test
Add the following test to a new test file called src/tests/test_notes.py:

import json
import pytest
from app.api import crud

def test_create_note(test_app, monkeypatch):
    test_request_payload = {"title": "something", "description": "something else"}
    test_response_payload = {"id": 1, "title": "something", "description": "something else"}

    async def mock_post(payload):
        return 1

    monkeypatch.setattr(crud, "post", mock_post)

    response = test_app.post("/notes/", data=json.dumps(test_request_payload),)

    assert response.status_code == 201
    assert response.json() == test_response_payload

def test_create_note_invalid_json(test_app):
    response = test_app.post("/notes/", data=json.dumps({"title": "something"}))
    assert response.status_code == 422
This test uses the Pytest monkeypatch fixture to mock out the crud.post function. We then asserted that the endpoint responds with the expected status codes and response body.

$ docker-compose exec web pytest .

========================================== test session starts ==========================================
platform linux -- Python 3.8.1, pytest-5.4.1, py-1.8.1, pluggy-0.13.1
rootdir: /usr/src/app
collected 3 items                                                                                       

tests/test_notes.py ..                                                                            [ 66%]
tests/test_ping.py .                                                                              [100%]

=========================================== 3 passed in 0.05s ===========================================
Now we can configure the remaining CRUD routes using Test-Driven Development.

fastapi
    ├── docker-compose.yml
    └── src
        ├── Dockerfile
        ├── app
        │   ├── __init__.py
        │   ├── api
        │   │   ├── __init__.py
        │   │   ├── crud.py
        │   │   ├── models.py
        │   │   ├── notes.py
        │   │   └── ping.py
        │   ├── db.py
        │   └── main.py
        ├── requirements.txt
        └── tests
            ├── __init__.py
            ├── conftest.py
            ├── test_notes.py
            └── test_ping.py
GET routes
GET one note
Test
Add the following tests (inside test_notes.py):

def test_read_note(test_app, monkeypatch):
    test_data = {"id": 1, "title": "something", "description": "something else"}

    async def mock_get(id):
        return test_data

    monkeypatch.setattr(crud, "get", mock_get)

    response = test_app.get("/notes/1")
    assert response.status_code == 200
    assert response.json() == test_data

def test_read_note_incorrect_id(test_app, monkeypatch):
    async def mock_get(id):
        return None

    monkeypatch.setattr(crud, "get", mock_get)

    response = test_app.get("/notes/999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Note not found"
Run tests: docker-compose exec web pytest .

They should fail:

========================================== test session starts ==========================================
platform linux -- Python 3.8.1, pytest-5.4.1, py-1.8.1, pluggy-0.13.1
rootdir: /usr/src/app
collected 3 items                                                                                       

tests/test_notes.py ..                                                                            [ 66%]
tests/test_ping.py .                                                                              [100%]

=========================================== 3 passed in 0.05s ===========================================
(base) rhkina@rhkina-ThinkPad-T440:~/Workspace/fastapi$ docker-compose exec web pytest .
========================================== test session starts ==========================================
platform linux -- Python 3.8.1, pytest-5.4.1, py-1.8.1, pluggy-0.13.1
rootdir: /usr/src/app
collected 5 items                                                                                       

tests/test_notes.py ..FF                                                                          [ 80%]
tests/test_ping.py .                                                                              [100%]

=============================================== FAILURES ================================================
____________________________________________ test_read_note _____________________________________________

test_app = <starlette.testclient.TestClient object at 0x7f8c5297e8b0>
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f8c5297e6d0>

    def test_read_note(test_app, monkeypatch):
        test_data = {"id": 1, "title": "something", "description": "something else"}

        async def mock_get(id):
            return test_data

>       monkeypatch.setattr(crud, "get", mock_get)
E       AttributeError: <module 'app.api.crud' from '/usr/src/app/app/api/crud.py'> has no attribute 'get'

tests/test_notes.py:29: AttributeError
______________________________________ test_read_note_incorrect_id ______________________________________

test_app = <starlette.testclient.TestClient object at 0x7f8c5297e8b0>
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f8c52615e20>

    def test_read_note_incorrect_id(test_app, monkeypatch):
        async def mock_get(id):
            return None

>       monkeypatch.setattr(crud, "get", mock_get)
E       AttributeError: <module 'app.api.crud' from '/usr/src/app/app/api/crud.py'> has no attribute 'get'

tests/test_notes.py:39: AttributeError
======================================== short test summary info ========================================
FAILED tests/test_notes.py::test_read_note - AttributeError: <module 'app.api.crud' from '/usr/src/app...
FAILED tests/test_notes.py::test_read_note_incorrect_id - AttributeError: <module 'app.api.crud' from ...
====================================== 2 failed, 3 passed in 0.13s ======================================
Code
Add the handler (inside notes.py):

@router.get("/{id}/", response_model=NoteDB)
async def read_note(id: int):
    note = await crud.get(id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return note
Here, instead of taking a payload, the handler requires an id, an integer, which will come from the path -- i.e., /notes/5/.

Add the get utility function to crud.py:

async def get(id: int):
    query = notes.select().where(id == notes.c.id)
    return await database.fetch_one(query=query)
Before moving on, ensure the tests pass and manually test the new endpoint in the browser, with curl or HTTPie, and/or via the API documentation.

GET all notes
Test
Next, add a test for reading all notes:

def test_read_all_notes(test_app, monkeypatch):
    test_data = [
        {"title": "something", "description": "something else", "id": 1},
        {"title": "someone", "description": "someone else", "id": 2},
    ]

    async def mock_get_all():
        return test_data

    monkeypatch.setattr(crud, "get_all", mock_get_all)

    response = test_app.get("/notes/")
    assert response.status_code == 200
    assert response.json() == test_data
Again, make sure the test fails. docker-compose exec web pytest .

========================================== test session starts ==========================================
platform linux -- Python 3.8.1, pytest-5.4.1, py-1.8.1, pluggy-0.13.1
rootdir: /usr/src/app
collected 6 items                                                                                       

tests/test_notes.py ....F                                                                         [ 83%]
tests/test_ping.py .                                                                              [100%]

=============================================== FAILURES ================================================
__________________________________________ test_read_all_notes __________________________________________

test_app = <starlette.testclient.TestClient object at 0x7fefa6ffa8e0>
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7fefa6ffadf0>

    def test_read_all_notes(test_app, monkeypatch):
        test_data = [
            {"title": "something", "description": "something else", "id": 1},
            {"title": "someone", "description": "someone else", "id": 2},
        ]

        async def mock_get_all():
            return test_data

>       monkeypatch.setattr(crud, "get_all", mock_get_all)
E       AttributeError: <module 'app.api.crud' from '/usr/src/app/app/api/crud.py'> has no attribute 'get_all'

tests/test_notes.py:54: AttributeError
======================================== short test summary info ========================================
FAILED tests/test_notes.py::test_read_all_notes - AttributeError: <module 'app.api.crud' from '/usr/sr...
====================================== 1 failed, 5 passed in 0.14s ======================================
Code
Handler Add the following code to notes.py:

@router.get("/", response_model=List[NoteDB])
async def read_all_notes():
    return await crud.get_all()
And import List from Python's typing module (include at the top of the notes.py): from typing import List

The response_model is a List with a NoteDB subtype.

Util Add the CRUD util at crud.py:

async def get_all():
    query = notes.select()
    return await database.fetch_all(query=query)
Make sure the automated tests pass now. Manually test this endpoint as well. docker-compose exec web pytest .

========================================== test session starts ==========================================
platform linux -- Python 3.8.1, pytest-5.4.1, py-1.8.1, pluggy-0.13.1
rootdir: /usr/src/app
collected 6 items                                                                                       

tests/test_notes.py .....                                                                         [ 83%]
tests/test_ping.py .                                                                              [100%]

=========================================== 6 passed in 0.08s ===========================================
PUT route
Test
Add the code to the test_notes.py:

def test_update_note(test_app, monkeypatch):
    test_update_data = {"title": "someone", "description": "someone else", "id": 1}

    async def mock_get(id):
        return True

    monkeypatch.setattr(crud, "get", mock_get)

    async def mock_put(id, payload):
        return 1

    monkeypatch.setattr(crud, "put", mock_put)

    response = test_app.put("/notes/1/", data=json.dumps(test_update_data))
    assert response.status_code == 200
    assert response.json() == test_update_data

@pytest.mark.parametrize(
    "id, payload, status_code",
    [
        [1, {}, 422],
        [1, {"description": "bar"}, 422],
        [999, {"title": "foo", "description": "bar"}, 404],
    ],
)
def test_update_note_invalid(test_app, monkeypatch, id, payload, status_code):
    async def mock_get(id):
        return None

    monkeypatch.setattr(crud, "get", mock_get)

    response = test_app.put(f"/notes/{id}/", data=json.dumps(payload),)
    assert response.status_code == status_code
This test uses the Pytest parametrize decorator to parametrize the arguments for the test_update_note_invalid function.

Run tests: $ docker-compose exec web pytest .

========================================== test session starts ==========================================
platform linux -- Python 3.8.1, pytest-5.4.1, py-1.8.1, pluggy-0.13.1
rootdir: /usr/src/app
collected 10 items                                                                                      

tests/test_notes.py .....FFFF                                                                     [ 90%]
tests/test_ping.py .                                                                              [100%]

=============================================== FAILURES ================================================
___________________________________________ test_update_note ____________________________________________

test_app = <starlette.testclient.TestClient object at 0x7f0489984a90>
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f04899846a0>

    def test_update_note(test_app, monkeypatch):
        test_update_data = {"title": "someone", "description": "someone else", "id": 1}

        async def mock_get(id):
            return True

        monkeypatch.setattr(crud, "get", mock_get)

        async def mock_put(id, payload):
            return 1

>       monkeypatch.setattr(crud, "put", mock_put)
E       AttributeError: <module 'app.api.crud' from '/usr/src/app/app/api/crud.py'> has no attribute 'put'

tests/test_notes.py:71: AttributeError
_______________________________ test_update_note_invalid[1-payload0-422] ________________________________

test_app = <starlette.testclient.TestClient object at 0x7f0489984a90>
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f0489976100>, id = 1, payload = {}
status_code = 422

    @pytest.mark.parametrize(
        "id, payload, status_code",
        [
            [1, {}, 422],
            [1, {"description": "bar"}, 422],
            [999, {"title": "foo", "description": "bar"}, 404],
        ],
    )
    def test_update_note_invalid(test_app, monkeypatch, id, payload, status_code):
        async def mock_get(id):
            return None

        monkeypatch.setattr(crud, "get", mock_get)

        response = test_app.put(f"/notes/{id}/", data=json.dumps(payload),)
>       assert response.status_code == status_code
E       assert 405 == 422
E        +  where 405 = <Response [405]>.status_code

tests/test_notes.py:93: AssertionError
_______________________________ test_update_note_invalid[1-payload1-422] ________________________________

test_app = <starlette.testclient.TestClient object at 0x7f0489984a90>
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f04898577c0>, id = 1
payload = {'description': 'bar'}, status_code = 422

    @pytest.mark.parametrize(
        "id, payload, status_code",
        [
            [1, {}, 422],
            [1, {"description": "bar"}, 422],
            [999, {"title": "foo", "description": "bar"}, 404],
        ],
    )
    def test_update_note_invalid(test_app, monkeypatch, id, payload, status_code):
        async def mock_get(id):
            return None

        monkeypatch.setattr(crud, "get", mock_get)

        response = test_app.put(f"/notes/{id}/", data=json.dumps(payload),)
>       assert response.status_code == status_code
E       assert 405 == 422
E        +  where 405 = <Response [405]>.status_code

tests/test_notes.py:93: AssertionError
______________________________ test_update_note_invalid[999-payload2-404] _______________________________

test_app = <starlette.testclient.TestClient object at 0x7f0489984a90>
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f04898666d0>, id = 999
payload = {'description': 'bar', 'title': 'foo'}, status_code = 404

    @pytest.mark.parametrize(
        "id, payload, status_code",
        [
            [1, {}, 422],
            [1, {"description": "bar"}, 422],
            [999, {"title": "foo", "description": "bar"}, 404],
        ],
    )
    def test_update_note_invalid(test_app, monkeypatch, id, payload, status_code):
        async def mock_get(id):
            return None

        monkeypatch.setattr(crud, "get", mock_get)

        response = test_app.put(f"/notes/{id}/", data=json.dumps(payload),)
>       assert response.status_code == status_code
E       assert 405 == 404
E        +  where 405 = <Response [405]>.status_code

tests/test_notes.py:93: AssertionError
======================================== short test summary info ========================================
FAILED tests/test_notes.py::test_update_note - AttributeError: <module 'app.api.crud' from '/usr/src/a...
FAILED tests/test_notes.py::test_update_note_invalid[1-payload0-422] - assert 405 == 422
FAILED tests/test_notes.py::test_update_note_invalid[1-payload1-422] - assert 405 == 422
FAILED tests/test_notes.py::test_update_note_invalid[999-payload2-404] - assert 405 == 404
====================================== 4 failed, 6 passed in 0.21s ======================================
Code
Handler (notes.py)

@router.put("/{id}/", response_model=NoteDB)
async def update_note(id: int, payload: NoteSchema):
    note = await crud.get(id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    note_id = await crud.put(id, payload)

    response_object = {
        "id": note_id,
        "title": payload.title,
        "description": payload.description,
    }
    return response_object
Util (crud.py)

async def put(id: int, payload: NoteSchema):
    query = (
        notes
        .update()
        .where(id == notes.c.id)
        .values(title=payload.title, description=payload.description)
        .returning(notes.c.id)
    )
    return await database.execute(query=query)

Run tests: $ docker-compose exec web pytest .

========================================== test session starts ==========================================
platform linux -- Python 3.8.1, pytest-5.4.1, py-1.8.1, pluggy-0.13.1
rootdir: /usr/src/app
collected 10 items                                                                                      

tests/test_notes.py .........                                                                     [ 90%]
tests/test_ping.py .                                                                              [100%]

========================================== 10 passed in 0.12s ===========================================
Manually test this endpoint as well.

DELETE route
Test
In test_notes.py add:

def test_remove_note(test_app, monkeypatch):
    test_data = {"title": "something", "description": "something else", "id": 1}

    async def mock_get(id):
        return test_data

    monkeypatch.setattr(crud, "get", mock_get)

    async def mock_delete(id):
        return id

    monkeypatch.setattr(crud, "delete", mock_delete)

    response = test_app.delete("/notes/1/")
    assert response.status_code == 200
    assert response.json() == test_data


def test_remove_note_incorrect_id(test_app, monkeypatch):
    async def mock_get(id):
        return None

    monkeypatch.setattr(crud, "get", mock_get)

    response = test_app.delete("/notes/999/")
    assert response.status_code == 404
    assert response.json()["detail"] == "Note not found"
Run tests and make sure you get errors: $ docker-compose exec web pytest .

========================================== test session starts ==========================================
platform linux -- Python 3.8.1, pytest-5.4.1, py-1.8.1, pluggy-0.13.1
rootdir: /usr/src/app
collected 12 items                                                                                      

tests/test_notes.py .........FF                                                                   [ 91%]
tests/test_ping.py .                                                                              [100%]

=============================================== FAILURES ================================================
___________________________________________ test_remove_note ____________________________________________

test_app = <starlette.testclient.TestClient object at 0x7f30ec5cb730>
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f30ec5900a0>

    def test_remove_note(test_app, monkeypatch):
        test_data = {"title": "something", "description": "something else", "id": 1}

        async def mock_get(id):
            return test_data

        monkeypatch.setattr(crud, "get", mock_get)

        async def mock_delete(id):
            return id

>       monkeypatch.setattr(crud, "delete", mock_delete)
E       AttributeError: <module 'app.api.crud' from '/usr/src/app/app/api/crud.py'> has no attribute 'delete'

tests/test_notes.py:106: AttributeError
_____________________________________ test_remove_note_incorrect_id _____________________________________

test_app = <starlette.testclient.TestClient object at 0x7f30ec5cb730>
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f30ec5da970>

    def test_remove_note_incorrect_id(test_app, monkeypatch):
        async def mock_get(id):
            return None

        monkeypatch.setattr(crud, "get", mock_get)

        response = test_app.delete("/notes/999/")
>       assert response.status_code == 404
E       assert 405 == 404
E        +  where 405 = <Response [405]>.status_code

tests/test_notes.py:120: AssertionError
======================================== short test summary info ========================================
FAILED tests/test_notes.py::test_remove_note - AttributeError: <module 'app.api.crud' from '/usr/src/a...
FAILED tests/test_notes.py::test_remove_note_incorrect_id - assert 405 == 404
===================================== 2 failed, 10 passed in 0.19s ======================================
Code
Handler (notes.py)

@router.delete("/{id}/", response_model=NoteDB)
async def delete_note(id: int):
    note = await crud.get(id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    await crud.delete(id)

    return note
Util (crud.py)

async def delete(id: int):
    query = notes.delete().where(id == notes.c.id)
    return await database.execute(query=query)
Make sure all tests pass: $ docker-compose exec web pytest .

========================================== test session starts ==========================================
platform linux -- Python 3.8.1, pytest-5.4.1, py-1.8.1, pluggy-0.13.1
rootdir: /usr/src/app
collected 12 items                                                                                      

tests/test_notes.py ...........                                                                   [ 91%]
tests/test_ping.py .                                                                              [100%]

========================================== 12 passed in 0.13s ===========================================
Additional Validation
Let's add some additional validation to routes, checking:

The id is greater than 0 for reading a single note, updating a note, and deleting a note.
The title and description fields from the request payloads must have lengths >= 3 and <= 50 for adding and updating a note.
GET
Update the test_read_note_incorrect_id test in test_notes.py:

def test_read_note_incorrect_id(test_app, monkeypatch):
    async def mock_get(id):
        return None

    monkeypatch.setattr(crud, "get", mock_get)

    response = test_app.get("/notes/999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Note not found"

    response = test_app.get("/notes/0")
    assert response.status_code == 422
The test should fail: docker-compose exec web pytest .

========================================== test session starts ==========================================
platform linux -- Python 3.8.1, pytest-5.4.1, py-1.8.1, pluggy-0.13.1
rootdir: /usr/src/app
collected 12 items                                                                                      

tests/test_notes.py ...F.......                                                                   [ 91%]
tests/test_ping.py .                                                                              [100%]

=============================================== FAILURES ================================================
______________________________________ test_read_note_incorrect_id ______________________________________

test_app = <starlette.testclient.TestClient object at 0x7f5d02bfac10>
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f5d02bfa280>

    def test_read_note_incorrect_id(test_app, monkeypatch):
        async def mock_get(id):
            return None

        monkeypatch.setattr(crud, "get", mock_get)

        response = test_app.get("/notes/999")
        assert response.status_code == 404
        assert response.json()["detail"] == "Note not found"

        response = test_app.get("/notes/0")
>       assert response.status_code == 422
E       assert 404 == 422
E        +  where 404 = <Response [404]>.status_code

tests/test_notes.py:46: AssertionError
======================================== short test summary info ========================================
FAILED tests/test_notes.py::test_read_note_incorrect_id - assert 404 == 422
===================================== 1 failed, 11 passed in 0.19s ======================================
Update the handler (in notes.py):

@router.get("/{id}/", response_model=NoteDB)
async def read_note(id: int = Path(..., gt=0),):
    note = await crud.get(id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return note
Make sure to import Path (at the top of notes.py); from fastapi import APIRouter, HTTPException, Path

So, we added the following metadata to the parameter with Path: 1. ... - the value is required (Ellipsis) 2. gt - the value must be greater than 0

The tests should pass: $ docker-compose exec web pytest .

========================================== test session starts ==========================================
platform linux -- Python 3.8.1, pytest-5.4.1, py-1.8.1, pluggy-0.13.1
rootdir: /usr/src/app
collected 12 items                                                                                      

tests/test_notes.py ...........                                                                   [ 91%]
tests/test_ping.py .                                                                              [100%]

========================================== 12 passed in 0.13s ===========================================
Try out the API documentation as well:

422 error

POST
Update the test_create_note_invalid_json test (test_notes.py):

def test_create_note_invalid_json(test_app):
    response = test_app.post("/notes/", data=json.dumps({"title": "something"}))
    assert response.status_code == 422

    response = test_app.post("/notes/", data=json.dumps({"title": "1", "description": "2"}))
    assert response.status_code == 422
Run the test and you should see the error (now on, I will not show the command and the result).

To get the test to pass, update the NoteSchema (in models.py) like so:

class NoteSchema(BaseModel):
    title: str = Field(..., min_length=3, max_length=50)
    description: str = Field(..., min_length=3, max_length=50)
As we added additional validation to the pydantic model with Field, add the import (in the same models.py file): from pydantic import BaseModel, Field

PUT
Add three more scenarios to test_update_note_invalid (test_notes.py):

@pytest.mark.parametrize(
    "id, payload, status_code",
    [
        [1, {}, 422],
        [1, {"description": "bar"}, 422],
        [999, {"title": "foo", "description": "bar"}, 404],
        [1, {"title": "1", "description": "bar"}, 422],
        [1, {"title": "foo", "description": "1"}, 422],
        [0, {"title": "foo", "description": "bar"}, 422],
    ],
)
def test_update_note_invalid(test_app, monkeypatch, id, payload, status_code):
    async def mock_get(id):
        return None

    monkeypatch.setattr(crud, "get", mock_get)

    response = test_app.put(f"/notes/{id}/", data=json.dumps(payload),)
    assert response.status_code == status_code
Handler
notes.py:

@router.put("/{id}/", response_model=NoteDB)
async def update_note(payload: NoteSchema, id: int = Path(..., gt=0),):
    note = await crud.get(id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    note_id = await crud.put(id, payload)

    response_object = {
        "id": note_id,
        "title": payload.title,
        "description": payload.description,
    }
    return response_object
DELETE
Test (test_notes.py):

def test_remove_note_incorrect_id(test_app, monkeypatch):
    async def mock_get(id):
        return None

    monkeypatch.setattr(crud, "get", mock_get)

    response = test_app.delete("/notes/999/")
    assert response.status_code == 404
    assert response.json()["detail"] == "Note not found"

    response = test_app.delete("/notes/0/")
    assert response.status_code == 422
Test Error:

========================================== test session starts ==========================================
platform linux -- Python 3.8.1, pytest-5.4.1, py-1.8.1, pluggy-0.13.1
rootdir: /usr/src/app
collected 15 items                                                                                      

tests/test_notes.py .............F                                                                [ 93%]
tests/test_ping.py .                                                                              [100%]

=============================================== FAILURES ================================================
_____________________________________ test_remove_note_incorrect_id _____________________________________

test_app = <starlette.testclient.TestClient object at 0x7ff9374014f0>
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7ff9373b4460>

    def test_remove_note_incorrect_id(test_app, monkeypatch):
        async def mock_get(id):
            return None

        monkeypatch.setattr(crud, "get", mock_get)

        response = test_app.delete("/notes/999/")
        assert response.status_code == 404
        assert response.json()["detail"] == "Note not found"

        response = test_app.delete("/notes/0/")
>       assert response.status_code == 422
E       assert 404 == 422
E        +  where 404 = <Response [404]>.status_code

tests/test_notes.py:133: AssertionError
======================================== short test summary info ========================================
FAILED tests/test_notes.py::test_remove_note_incorrect_id - assert 404 == 422
===================================== 1 failed, 14 passed in 0.23s ======================================
Handler (notes.py):

@router.delete("/{id}/", response_model=NoteDB)
async def delete_note(id: int = Path(..., gt=0)):
    note = await crud.get(id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")

    await crud.delete(id)

    return note
The tests should pass:

========================================== test session starts ==========================================
platform linux -- Python 3.8.1, pytest-5.4.1, py-1.8.1, pluggy-0.13.1
rootdir: /usr/src/app
collected 15 items                                                                                      

tests/test_notes.py ..............                                                                [ 93%]
tests/test_ping.py .                                                                              [100%]

========================================== 15 passed in 0.16s ===========================================
Conclusion
FastAPI is an awesome asynchronous Python micro framework! With Flask-like simplicity, it is easy and fun to update your Flask RESTful API to asynchronous mode. Hope you have enjoyed!

References
Original tutorial:
Herman, Michael - Developing and Testing an Asynchronous API with FastAPI and Pytest

Other references:
FastAPI Official Documentation - https://fastapi.tiangolo.com/

Uvicorn Official Documentation - https://fastapi.tiangolo.com/

Starlette Official Documentation - https://www.starlette.io/

Pydantic Official Documentation - https://pydantic-docs.helpmanual.io/

Built with MkDocs using a theme provided by Read the Docs.