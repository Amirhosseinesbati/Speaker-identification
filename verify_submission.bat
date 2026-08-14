@echo off
REM ============================================================
REM  Verify submission_leaderboard.zip - full leaderboard replay
REM  Run this by double-clicking, or paste this one line in cmd:
REM     verify_submission.bat
REM ============================================================

"D:\Projects\My projects\IAAA_Compet\leaderbordvenv\.venv\Scripts\python.exe" "D:\Projects\My projects\IAAA_Compet\Speaker-identification\scripts\verify_submission.py"

echo.
echo Exit code: %ERRORLEVEL%
if "%ERRORLEVEL%"=="0" (
    echo ALL CHECKS PASSED - zip is ready to submit.
) else (
    echo VERIFICATION FAILED - see output above.
)
pause
