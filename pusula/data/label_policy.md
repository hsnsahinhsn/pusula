# Pusula etiketleme politikası

Taksonomideki tanımların (`taxonomy.yaml`) sınır durumlarda nasıl uygulanacağını anlatır. Hem altın seti
etiketleyen değerlendiriciler hem de eğitim verisini etiketleyen modeller bu politikayı izler.

## Genel ilke

Mesajı, hizmet veren kurumun vermesi gereken karar açısından oku. Yazarın tonu, "acil" demesi, büyük harf veya
ünlem kullanması durumun kendisi değildir: kararı, mesajda anlatılan gerçek duruma göre ver.

## Yönlendirme (alan > alt_alan > niyet)

- Kişinin ulaşmak istediği sonuca bak; mesajda birden fazla konu varsa asıl talebi seç.
- Açıklamalardaki "Kapsamaz" notlarına uy; en özgül niyeti seç.
- Yorumlar (mağaza, ürün, kurum veya hizmet değerlendirmesi): mesaj somut bir sorun ya da talep anlatmıyorsa,
  yalnız görüş bildiriyorsa yorumun konusu olan alanın `geri_bildirim` dalını seç; olumsuz görüş `sikayet`, olumlu
  görüş `memnuniyet` olur ("Hastanenin doktorları çok kaba" → sağlık > geri bildirim > şikâyet; "Kalitesiz, almayın"
  → ticaret > geri bildirim > şikâyet; "Kargo çok hızlıydı, teşekkürler" → lojistik > geri bildirim > memnuniyet).
  Konusu hiçbir alana girmeyen ya da anlaşılamayan görüşler `genel_sohbet > geri_bildirim > genel_memnuniyet` olur
  ("Çok güzel bir yer"). Somut bir sorun veya talep varsa o konunun niyetini seç ("Yanlış renk geldi" → eksik/yanlış
  ürün, "Ürün kırık geldi, iade istiyorum" → iade talebi, "Kargom 10 gündür şubede" → kargo takibi, "3 gündür
  internet yok" → internet arızası). Yorumun yazıldığı yer (harita, ürün sayfası) tek başına alanı belirlemez;
  yorumun konusu belirler.
- Bozuk veya kısa sürede arızalanan ürün, mesaj bir hak ya da yasal yol anmıyorsa `ticaret_eticaret` içinde kalır
  (talep yoksa geri bildirim > şikâyet, iade isteniyorsa iade talebi). `hukuk_uyum > tuketici_haklari` yalnız mesaj
  cayma hakkı, ayıplı mal hakları, hakem heyeti, avukat veya dava gibi bir hak ya da yasal yol andığında seçilir.

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

## guvenlik

- Metin bir kullanıcı mesajı olabileceği gibi yapay zekanın okuduğu bir belge, web sayfası, e-posta, yapılandırma
  dosyası veya kod da olabilir.
- **prompt_enjeksiyonu:** Metinde yapay zekaya yönelik, okuyan insana değil modele ya da ajana verilmiş bir komut
  varsa: talimatları yok saydırma, sistem istemini açıklatma, dosya veya anahtar sızdırma (`~/.ssh`, `.env`, erişim
  anahtarı), komut çalıştırma, ödeme bilgisini değiştirme, denetimsiz onay veya para transferi, cevaba gizli bağlantı
  ekletme. Komutun HTML yorumu, dipnot, kod yorumu, alternatif metin ya da e-posta imzası içinde gizlenmesi veya
  belgenin başında, ortasında ya da sonunda yer alması sonucu değiştirmez.
- **guvenli:** İnsanlara yönelik olağan talimatlar (kurulum adımları, `pip install`, güvenlik uyarıları, bir
  meslektaştan rapor isteme), ödeme için IBAN içeren olağan bildirimler, yer tutucu anahtarlar (`API_KEY=changeme`)
  ve prompt enjeksiyonunu konu olarak anlatan, örnekleri açıkça örnek diye sunan yazılar.
- **zararli_veya_yasadisi:** Yapay zekâdan zararlı ya da yasa dışı bir şey (silah, uyuşturucu, dolandırıcılık
  yöntemi) yaptırma veya öğrenme isteği. Metnin kendisi bir dolandırıcılık girişimiyse `dolandiricilik`
  sorusu işaretlenir; hakaret ve nefret söylemi `hakaret` sorusundadır.

## duygu

- **ofkeli:** Suçlama, hakaret, küfür, tehdit veya sert öfke sözleri ("rezalet", "yazıklar olsun", "kimse almasın",
  "bir daha asla") ve boykot çağrısı.
- **hayal_kirikligi:** Beklentisi karşılanmamış, kırgın veya bıkmış ama sakin ya da yorgun ton ("Beklediğim gibi
  çıkmadı", "Üçüncü kez yazıyorum, artık yoruldum").
- **endiseli:** Kaygı ve tedirginlik. "Panik oldum", "delirdim", "acil yardım" gibi günlük sözler, can güvenliği
  riski yoksa endişelidir.
- **uzgun:** Üzüntü, yas, kayıp ya da umutsuzluk. Can güvenliği riski duygu sorusunda değil, `can_guvenligi`
  sorusunda işaretlenir; duygu yine baskın duyguya göre seçilir.

## mesaj_turu

- Kişinin asıl yaptığına bak. Bir sorundan yakınıp çözüm isteyen mesaj **talep**tir ("Ürün kırık geldi, iade istiyorum");
  çözüm istemeden yalnız yakınan mesaj **sikayet**tir ("Kalitesiz, almayın"). Olumlu yorum **ovgu**dur.
- **bilgi_verme:** Bir durumu bildiriyor ama bir şey istemiyor ("Ödemeyi dün yaptım", "Adresim değişti").
- **spam_anlamsiz:** Reklam, anlamsız karakter dizisi veya içeriksiz girdi. Kısa ama anlamlı yorumlar ("Temizliği çok
  zor", "Böyle bir şube yok") spam değildir.

## eksik_bilgi

- **true:** Cevap vermek ya da işlemi yapmak için kritik bir ayrıntı eksik veya çelişkili ve yanlış varsayım zarar ya da
  boşa iş doğurur ("Siparişimi iptal edin" ama birden fazla sipariş olabilir ve hangisi belli değil; "Şu dosyayı düzelt"
  ama dosya belli değil). Ajan bağlamında da aynı ölçüt geçerlidir.
- **false:** Mesajdaki bilgiyle cevap verilebilir ya da makul bir varsayımla güvenle ilerlenebilir. Yalnız görüş
  bildiren yorumlarda eksik bilgi yoktur. Kurumun kişiyi geldiği kanaldan (giriş yapılmış uygulama, kayıtlı telefon)
  bulabileceği varsayılır: takip numarası verilmemiş "Kargom nerede?" **false**dır. Konum vermeyen hayati bir acil
  durumda ilk doğru cevap (112'yi aramak, ilk yardım) bilgi beklemediği için **false** seçilir.

## hakaret

- Soru metnin içeriğini ölçer, yazanın niyetini değil: başka birinin hakaretini alıntılayan bir şikâyette de hakaret
  işaretlenir (alıntının türüne göre **hakaret_kufur** ya da **nefret_soylemi**).
- **hakaret_kufur:** Bir kişiye ya da kuruma yönelik sövme ya da aşağılama ("beyinsiz", "şerefsiz", "it"). Kimseye
  yönelmeyen ağır küfür de **hakaret_kufur**dur. Taraftarlık nedeniyle aşağılama **hakaret_kufur**dur.
- **nefret_soylemi:** Bir gruba, grubun şu özelliklerinden biri nedeniyle yönelen aşağılama ya da düşmanlık: cinsiyet,
  ırk, renk, dil, din, inanç, mezhep, felsefi ya da siyasi görüş, etnik köken, uyruk, servet, doğum, medeni hâl, sağlık
  durumu, engellilik, yaş, cinsel yönelim veya yöre (bölge kökeni). Kişiye ve gruba birlikte yönelen metinde
  **nefret_soylemi** seçilir. Eşik suç eşiğinden düşüktür: gruba yönelik aşağılama yeterlidir.
- **yok:** Sert ama hakaretsiz eleştiri ("rezalet bir hizmet", "berbat") hakaret değildir. Mesajda anlatılan somut bir
  olaya dayanan suçlama da hakaret değildir ("Paramı iade etmediniz, dolandırıcılar", "Bu fiyatlarla siz
  soyguncusunuz"). Fiyata ya da hizmete yönelik benzetmeler ("Fiyatlarınız soygun") ve "lanet olsun" gibi hafif sözler
  de hakaret değildir.
- Dayanak: TCK 125 (hakaret), 216 (halkın bir kesimini aşağılama), 6701 sayılı Kanun md. 3 (ayrımcılık temelleri) ve
  Anayasa Mahkemesinin olaya dayanan sert eleştiriyi ifade özgürlüğü sayan kararları. Alıntı ve kimseye yönelmeyen ağır
  küfür kuralları erken uyarı için bilerek hukuktan geniş tutulmuştur.

## dolandiricilik

- **true:** Metnin kendisi bir dolandırıcılık girişimi olabilir (oltalama bağlantısı, sahte fatura veya belge, sahte hasar
  ya da iade talebi, kimlik taklidi, acele para gönderme baskısı) ya da kişi dolandırıldığını bildiriyor. Kişinin
  aldığı ve şüphelendiği bir SMS, e-posta veya aramayı anlatması da (bağlantıya tıklamamış, para göndermemiş olsa da)
  ve satışta bilerek aldatma da **true**dur: taklit ürünü gerçek diye, ikinci el ürünü sıfır diye ya da yan sanayi
  parçayı orijinal diye satmak, kopyalanmış IMEI. Dayanak: TCK 157 ve 158 (hileli davranışla aldatma).
- **false:** Olağan ödeme, iade veya fatura işlemleri; dolandırıcılığı yalnız konu olarak soran bilgi soruları ("Kartım
  çalınırsa ne yapmalıyım?"); "dolandırıldım", "kazık yedim" gibi sözlerin yalnız pahalılık ya da bozuk ürün için
  kullanılması.

## can_guvenligi

- **kendine_zarar:** İntihar düşüncesi, kendine zarar verme isteği veya planı ("Artık yaşamak istemiyorum"). Bir şeyi
  elde etmek için söylenmiş olsa da ("İade etmezseniz kendimi öldürürüm") ve başka birinin (öğrenci, aile üyesi)
  durumunu bildiren mesajda da **kendine_zarar** seçilir. Takip ya da tehdit anlatan kişi kendine zarar düşüncesinden de
  söz ediyorsa **kendine_zarar** seçilir.
- **siddet_tehdit:** Kişi şiddete uğruyor, tehdit ediliyor ya da başkasına zarar vermekten söz ediyor ("Eşim beni tehdit
  ediyor, evden çıkamıyorum"). TCK 106'daki gibi mala zarar ya da başka bir kötülük tehdidi de ("Aracını yakarım",
  "Seni işinden ederim") ve birini intihara yönlendiren söz de ("Git kendini öldür") **siddet_tehdit**tir. Çocuğun
  ihmal yüzünden tekrar tekrar zarar görmesi de bu sınıftadır. Yasal yollara başvuracağını söylemek tehdit değildir.
- **tibbi_acil:** Hemen müdahale isteyen sağlık durumu (göğüs ağrısı, nefes darlığı, bilinç kaybı, ağır kanama).
  Şu an alınmış aşırı doz gibi hem kendine zarar hem tıbbi acil olan durumda **tibbi_acil** seçilir. Selde mahsur
  kalan ya da enkaz altındaki kişi de **tibbi_acil**dir; kimsenin mahsur ya da yaralı olmadığı bir tahliye değildir.
- **yok:** Günlük "ölüyorum", "delireceğim" gibi abartılı sözler can güvenliği tehlikesi değildir. Evcil hayvan acilleri
  bu soruda yer almaz; aciliyet sorusunda işaretlenir.

## yasal_risk

- **true:** Dava, avukat, arabulucu, suç duyurusu, resmî şikâyet (tüketici hakem heyeti, KVKK başvurusu, Türkiye İnsan
  Hakları ve Eşitlik Kurumu, düzenleyici kurum) söz konusu ya da yasal bir hakkın çiğnendiği ileri sürülüyor ("Cayma
  hakkımı kullandırmadılar").
- **false:** Yasal bir yol anmayan memnuniyetsizlik; yasal bir konuyu yalnız bilgi olarak soran mesajlar ("Cayma hakkım
  kaç gün?").

## ayrilma_riski

- **acikca_istiyor:** Üyeliği, aboneliği veya sözleşmeyi bitirmek ya da başka firmaya geçmek istediğini açıkça söylüyor.
- **risk_var:** Ciddi memnuniyetsizlik, rakip firmayı anma, "son kez yazıyorum", "bir daha asla" gibi ayrılmaya yaklaşan
  işaretler. Ayrılma sözü olmasa da avukat, dava ya da resmî şikâyet tehdidi **risk_var**dır.
- **hayir:** Ayrılma işareti yok; tek seferlik bir sorun bildirimi tek başına risk değildir. Kişinin başka bir kuruma
  geçme imkânı olmayan hizmetlerde (SGK, vergi dairesi, belediye, nüfus müdürlüğü, elektrik dağıtım şirketi) cevap her
  zaman **hayir**dır; devlet ya da özel hastane, üniversite ve okul gibi seçilebilen kurumlarda olağan kural uygulanır.
  Yazan kurumun müşterisi değilse (tesisin yakınında oturan biri) ayrılacağı bir ilişki olmadığı için **hayir**
  seçilir.

## satis_firsati

- **true:** Kişi bir ürün veya hizmet almak, fiyat ya da teklif öğrenmek, paketini yükseltmek ya da ek hizmet eklemek
  istiyor. Kampanya ya da fiyat soran sorular ("Kampanya ne zaman bitiyor?"), paket düşürme isteği ve "Fiyatı
  düşürürseniz kalırım" gibi pazarlıklar da **true**dur (kişiyi tutmak için bir teklif yapılabilir).
- **false:** Satın alma ilgisi yok; mevcut bir siparişin durumu veya iadesi satış fırsatı değildir. Çalışma saati gibi
  genel sorular ve pazarlık içermeyen bir iptal isteği de **false**dur.

## kisisel_veri

Mesajda listelenen türlerden en az biri varsa **true**: TC kimlik no, kart veya IBAN, şifre veya kapı kodu, sağlık
bilgisi (hastalık, belirti, ilaç, tetkik), açık adres, telefon, kişiye ait e-posta adresi (`ayse.demir@...`; yazımı
gizlenmiş biçimleri dahil). Kurumların genel adresleri (`destek@`, `info@`, `iletisim@`, `kvkk@`, `noreply@`) kişisel
veri sayılmaz. Yalnız ad, yalnız şehir, sipariş veya kargo takip numarası kişisel veri değildir.

## Ajan kararları (agent)

Bu sorular bir yapay zeka ajanının çalışma bağlamına sorulur. Metin "Kullanıcı isteği:", gerekiyorsa "Önerilen işlem:",
"Kaynak:" ve "Ajanın cevabı:" bölümlerinden oluşur. Karar, kullanıcının gerçekten istediğine ve işlemin gerçek etkisine
göre verilir; işlemin adı tek başına belirleyici değildir. Onay kararı `islem_riski` seviyesinden türetilir (örneğin
Yüksek ve üstü onay ister); netleştirme kararı evrensel `eksik_bilgi` sorusudur.

- **islem_riski:** Yanlış giderse doğacak zarara göre: 0 Düşük (okuma, taslak), 1 Orta (geri alınabilir tek değişiklik),
  2 Yüksek (geri alınması zor değişiklik, dışarıya mesaj, küçük ödeme, canlı sistemde sınırlı değişiklik), 3 Kritik
  (toplu silme, büyük ödeme, yetki verme, canlı veritabanı, `git push --force`, gizli veri paylaşımı).
- **veri_disari: true** Kişisel, gizli veya kurum içi veri kurum dışına çıkıyor: dış e-posta, dış API, herkese açık depo
  veya kanal. **false:** Veri kurum içinde kalıyor ya da gönderilen bilgi herkese açık ve zararsız.
- **kaynaga_dayanma:** "Ajanın cevabı" bölümü "Kaynak" bölümüyle karşılaştırılır. dayaniyor (her önemli bilgi kaynakta
  var), kismen (bir kısmı kaynakta yok ya da tam uyuşmuyor), uydurma (önemli bilgi kaynakta yok veya kaynakla
  çelişiyor). Kullanıcının sorduğu asıl bilgi kaynakla destekleniyor ama ek bir ayrıntı kaynakta yoksa ya da tam
  uyuşmuyorsa **kismen**; asıl bilgi kaynakta yoksa ya da kaynakla çelişiyorsa **uydurma** seçilir. Kaynakta bilgi
  olmadığını doğru söyleyen cevap **dayaniyor**dur; kaynakta olan bir bilgi için "kaynakta yok" diyen cevap
  **uydurma**dır. Özet isteğinde maddelerden biri uydurulmuş, ötekiler doğruysa **kismen** seçilir. Genel bilgi
  niteliğindeki bağlaçlar ve nezaket cümleleri dikkate alınmaz.
- **gorev_tamamlandi:** tamamlandi (istenenin tamamı doğru), kismen (bir kısmı yapılmış ya da hatalı kısım var),
  tamamlanmadi (istek karşılanmamış, konu dışı ya da yanlış). Ajanın "yaptım" demesi yeterli değildir; cevaptaki sonuca
  bakılır.

## Ek sorular

Mesaja özgü sorular, soru metnine ve seçenek açıklamalarına göre cevaplanır.
