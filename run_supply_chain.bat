@echo off
cd /d "C:\Users\vikra\Desktop\Supply Chain Algo"
"C:\ProgramData\anaconda3\python.exe" -u "automation\run_pipeline.py" >> "logs\scheduler_output.log" 2>&1
exit /b %ERRORLEVEL%