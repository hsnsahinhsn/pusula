@echo off
cd /d "%~dp0"

echo [1/3] Dosyalar ekleniyor...
git add .

echo [2/3] Paketleme yapiliyor...
git commit -m "update: sync" 2>nul

echo [3/3] GitHub'a gonderiliyor...
git push origin main

if %ERRORLEVEL% equ 0 (
    echo.
    echo ========================================
    echo   BASARILI: GitHub guncellendi.
    echo   Pencere 2 saniye icinde kapaniyor...
    echo ========================================
    timeout /t 2 /nobreak >nul
    exit /b 0
) else (
    echo.
    echo ========================================
    echo   HATA: Git push basarisiz oldu.
    echo ========================================
    pause
)
