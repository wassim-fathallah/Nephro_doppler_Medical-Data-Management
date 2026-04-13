@echo off
REM Install LangChain dependencies
echo Installing LangChain and dependencies...
python -m pip install -r requirements_langchain.txt -q

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ===============================================
    echo ✅ Dependencies installed successfully!
    echo ===============================================
    echo.
    echo Next: Run the training script
    echo python train_final_with_llm.py
    echo.
    pause
) else (
    echo.
    echo ❌ Installation failed!
    echo Try manually: pip install -r requirements_langchain.txt
    echo.
    pause
)
