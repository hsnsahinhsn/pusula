@echo off
chcp 65001 >nul
title Turkish Decision Model - GitHub Push

echo ========================================================
echo   Turkish Decision Model - GitHub Senkronizasyon Aracı
echo ========================================================
echo.

:: Scriptin çalıştığı klasöre geç
cd /d "%~dp0"

echo [1/3] Değişiklikler taranıyor (git add)...
git add .

echo.
echo [2/3] Değişiklikler paketleniyor (git commit)...
git commit -m "update: %date% %time% otomatik senkronizasyon" 2>nul
if %errorlevel% equ 0 (
    echo       Yeni değişiklikler başarıyla paketlendi.
) else (
    echo       Paketlenecek yeni yerel değişiklik bulunamadı, mevcut commitler gönderiliyor.
)

echo.
echo [3/3] GitHub'a yükleniyor (git push)...
git push origin main

echo.
if %errorlevel% equ 0 (
    echo ========================================================
    echo   ✓ BAŞARILI! Tüm dosyalar GitHub'a aktarıldı.
    echo   Pencere 2 saniye içinde otomatik kapatılıyor...
    echo ========================================================
    timeout /t 2 /nobreak >nul
    exit /b 0
) else (
    echo ========================================================
    echo   X HATA! Gönderim sırasında bir sorun oluştu.
    echo   Lütfen internet bağlantınızı ve GitHub yetkinizi kontrol edin.
    echo ========================================================
    echo.
    echo Hatayı inceleyebilmeniz için pencere açık bırakıldı:
    pause
)
