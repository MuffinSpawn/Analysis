@echo off
SET SCRIPT_PATH=%~dp0
'python -m unittest %SCRIPT_PATH%\..\test\family_tree\test_api.py
python -m unittest discover -v -s %SCRIPT_PATH%\..
