# syntax=docker/dockerfile:1
# build nicepipe executable using python image first
FROM python:3.9
WORKDIR /code
RUN pip install "poetry==1.2.0b1"
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi
RUN poe post-install
COPY . .
# rename to app in order to prevent name collision with the folder where the models are kept
RUN poe build-linux --onedir --name app

# same cuda version tested, modern distro to ensure glibc compatability
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04
COPY --from=0 /code/dist/app/ ./app/
COPY config.yml .
ENV LANG="C.UTF-8" LC_ALL="C.UTF-8"
EXPOSE 8000/tcp
EXPOSE 8000/udp
CMD ["./app/app"]


