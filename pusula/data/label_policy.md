# Pusula etiketleme politikası

Taksonomideki tanımların (`taxonomy.yaml`) sınır durumlarda nasıl uygulanacağını anlatır. Hem altın seti
etiketleyen değerlendiriciler hem de eğitim verisini etiketleyen modeller bu politikayı izler.

## Genel ilke

Mesajı, hizmet veren kurumun vermesi gereken karar açısından oku. Yazarın tonu, "acil" demesi, büyük harf veya
ünlem kullanması durumun kendisi değildir: kararı, mesajda anlatılan gerçek duruma göre ver.

## Yönlendirme (alan > alt_alan > niyet)

- Kişinin ulaşmak istediği sonuca bak; mesajda birden fazla konu varsa asıl talebi seç.
- Açıklamalardaki "Kapsamaz" notlarına uy; en özgül niyeti seç.

## aciliyet

| Seviye | Ne zaman | Örnek |
|---|---|---|
| 0 Rutin | Bilgi sorusu, öneri, teşekkür, planlı iş; beklemek kimseye zarar vermez | "Kargo ücretleriniz ne kadar?" |
| 1 Yakın | Sorun var ama iş sürüyor: yavaşlama, kısmi arıza, gecikme, iade bekleme, tekrarlayan şikâyet | "Depodaki yazıcı 2 saattir çalışmıyor, sipariş akışı yavaşladı" · "Paketim 2 haftadır gümrükte, 3. kez yazıyorum" |
| 2 Yüksek | İş veya hizmet tamamen durmuş ya da beklemek bugün somut zarar doğuracak | "3 gündür internet hiç yok, evden çalışıyorum" · "Maaş ödemesi askıda kaldı, bugün kira günü" |
| 3 Kritik | Can veya sağlık tehlikesi, süren dolandırıcılık ya da güvenlik ihlali, büyük maddi kayıp | "Kartımdan şu an bilgim dışında para çekiliyor" |

Israr, öfke veya "acil" kelimesi seviyeyi tek başına yükseltmez.

## eylem_gerekli

- **false:** Kişi bilgi soruyor (hakkım var mı, nasıl yapılır, ne zaman) veya yalnız görüş bildiriyor; cevap vermek yeterli.
  Örnek: "Bu sefer 4 saatlik gecikme için tazminat hakkım var mı?"
- **true:** Kişi açıkça bir işlem istiyor (iade, iptal, kayıt açma, bloke, değişiklik) ya da sorun ancak bir işlemle
  çözülebilir. Örnek: "Tazminat başvurumu başlatın" · "Yazıcı çalışmıyor" (servis kaydı gerekir).

## insana_devret

- **false:** Otomatik akış bilgi verip işlemi güvenle yürütebilir; ton duygusal olsa bile. Panik, öfke veya ısrar
  tek başına yetmez. Örnek: "Oğlum hafif bir kaza yaptı, panikteyim, önce sigortayı mı polisi mi aramalıyım?"
- **false:** Saldırı veya manipülasyon girişimleri (talimatları yok saydırma, gizli veri isteme) devredilmez;
  `guvenlik` sorusu bunları işaretler.
- **true:** Hukuki veya tıbbi risk, can güvenliği veya kendine zarar riski, dolandırıcılık mağduriyeti, yüksek tutar,
  yetki ya da istisna gerektiren karar, otomatik yanıtların çözemediği tekrar eden şikâyet.

## kisisel_veri

Mesajda listelenen türlerden en az biri varsa **true**: TC kimlik no, kart veya IBAN, şifre, sağlık bilgisi (hastalık,
belirti, ilaç, tetkik), açık adres, telefon.

## Ek sorular

Mesaja özgü sorular, soru metnine ve seçenek açıklamalarına göre cevaplanır.
