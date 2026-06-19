#### Install UV ###
## On Mac or Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

## Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

## Create an empty folder for the project and initialize it:
cd <Folder Name>

## Build projet folder
uv init

## Build virtual venv
uv venv

## Activate venv
source .venv/bin/activate