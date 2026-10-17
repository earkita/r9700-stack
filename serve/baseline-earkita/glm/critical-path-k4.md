# GLM decode K4: ślad i izolowany test down GEMV

## Fakty

- Źródło: `bench/results/glm-mtp-20261014/torchprof-dflash-diagnostic/`.
  Profil vLLM `e97573215`, C4 **pojemności** serwera, lecz jeden aktywny
  request (C1), 130816 tokenów wejścia i 64 wyjścia, DFlash K4 = cztery
  proponowane tokeny, `reasoning_effort=high`, temperatura 1.0, top-p 0.95,
  warm APC po jednym wyłączonym z pomiaru cold request. Odpowiedź zakończyła
  się limitem długości; ten ślad nie jest testem poprawności odpowiedzi.
- Procenty 24,9% W4A4 i 17,4% RCCL to średnia ośmiu ranków z **sum czasów
  kerneli GPU** w arbitralnym oknie 600–1600 ms od pierwszego kernela danego
  requestu. Nie jest to czas kroku, udział na ścieżce krytycznej ani prognoza
  przyspieszenia. W oknie GPU0: 188,2 ms `_gemv`, 132,8 ms RCCL z 759,4 ms
  sumy kerneli. Początek pełnego śladu zawiera prefill; okno zaczyna się po
  ostatnich zaobserwowanych kernelach prefillowych, ale nie ma znaczników
  `draft`/`verify`/`execute_model` potwierdzających dokładne granice kroków.
- Wspólna oś czasu ośmiu GPU jest możliwa: wszystkie pliki mają ten sam
  `baseTimeNanoseconds`, po 65514 kerneli i 2332 kernelowych epizodów NCCL;
  końce sparowanych epizodów są bliskie. W oknie 2051 epizodów ma medianę
  rozjazdu startów 9,52 µs (p90 18,95 µs, maksimum 45,80 µs), a rozjazdu
  końców 1,78 µs. W 99,4% par najwcześniej startujący rank ma dłuższy kernel
  niż najpóźniejszy. To **wspiera hipotezę oczekiwania na pozostałe ranki**;
  bez znaczników hosta i identyfikatorów operacji nie rozstrzyga przyczyny
  późniejszego wejścia. Epizod medianowy: 9,52 µs start / 0,65 µs koniec;
  największy rozjazd: 45,80 / 0,70 µs. Najdłuższy średni epizod trwał
  170,14 µs przy rozjeździe startów 8,04 µs — nie każdy długi kolektyw jest
  tylko oczekiwaniem.
- `_gemv` i NCCL na tym samym ranku nie nakładają się w śladzie. Czas NCCL
  na dowolnym ranku pokrywający `_gemv` innego ranku wynosi najwyżej 0,017 ms
  w całym sekundowym oknie. Łącznie przez 584,1 ms pracowało wszystkie osiem
  GPU; przez 44,4 ms żaden nie wykonywał kernela GPU, a w pozostałym czasie
  aktywna była część kart. Nie znamy z tego śladu udziału CPU/schedulera
  w przerwach ani czasu kroku. Nie dodajemy 25% i 17% jako możliwego zysku.
- W jednej powtarzalnej sekwencji ranku 0 `quant_dequant_mxfp4` → `_gemv`
  gate/up (~142 µs) → SILU/clamp (~3 µs) → `quant_dequant_mxfp4` → `_gemv`
  down (~85 µs) → `moe_sum` (~2 µs) → NCCL. W całym oknie: 817 długich
  `_gemv` = 117,5 ms, 816 krótkich = 70,7 ms; QDQ aktywacji 6,4 ms,
  aktywacja 2,7 ms, suma ekspertów 1,6 ms. Podział `_gemv` na gate/down jest
  wnioskiem z naprzemiennego porządku i czasu; ślad nie zawiera etykiet
  semantycznych. Kerneli top-k było 25,4 ms, ale nazwy nie pozwalają
  przypisać ich wszystkich do routera MoE, bo model ma też indexer.
- Dispatch: `r9700_vllm/compat/glm_moe.py:eligible/patch` kieruje zgodne
  wywołania GLM/gfx1201 do `r9700_vllm/moe/packed_gemv.py:apply/gemv/_gemv`;
  inne idą do upstream. Wejście BF16 `[M,4096]`, top-8 ekspertów, wagi
  uint8 z dwoma E2M1 na bajt, skale E8M0 uint8 na 32 wartości; gate/up
  `[288,512,2048]` oznacza N=512, K=4096, down `[288,4096,128]`
  N=4096, K=256. Aktywacje są QDQ MXFP4, a wynik po akumulacji FP32
  zapisywany BF16. Domyślny packed `BN=8`, `BK=min(256,K/2)`, 4 warpy;
  grid ma `M*8` tras. Weryfikacja K4 daje do 5 wierszy dla C1 i do 20
  dla C4; koszt zależy od tras i paddingu, którego dokładnej liczby dla
  śladu nie zapisano. Ślad C1 nie charakteryzuje kosztu C4.
- Bieżący launcher `glm-5.3-flash-c4-vllm771.sh` ma TP8, max-num-seqs=4,
  DFlash K4 TP8, FP8 KV, grafy FULL_DECODE_ONLY, `R9K_AG=0`, 16 KiB limitu
  własnego TP8 all-reduce. `--disable-custom-all-reduce` wyłącza backend
  vLLM, ale plugin instaluje własny P2P dla małych wiadomości. W większych
  przypadkach `R9kCommunicator` wraca do PyNCCL/RCCL. Implementacja modelu
  zawiera row-parallel projections, opcjonalne SP gather/scatter i DFlash
  `sp_all_gather`; sam kernel `ncclDevKernel_Generic_4` **nie ujawnia**
  op/dtype/numel. Nie przypisujemy wszystkich 2051 wywołań do all-reduce,
  do przepustowości PCIe ani do jednego typu wiadomości.
- Poprzedni A–B–A BN8→BN32→BN8, grupowany W4A4 i limit AR8 16→256 KiB
  nie potwierdziły end-to-end zysku C4. Jeden izolowany test gate/down
  wykazał BN64/4-warpy dla down 252,0 µs wobec BN8/4-warpy 329,2 µs przy
  20 wierszach; to **mikrokernel**, nie decode. Istniejący opt-in `wide`
  zmienia równocześnie gate i down, więc nie rozdzieli ich wkładu.

Powtarzalna analiza offline: `python3 profiling/glm_trace_critical_path.py
bench/results/glm-mtp-20261014/torchprof-dflash-diagnostic/traces`.
Wynik JSON: `.../critical-path-offline.json` (ignorowany artefakt).

## Brakujące obserwacje i hipotezy

Do rzeczywistej ścieżki krytycznej potrzebne są znaczniki kroków DFlash
(draft/verify/accept), host→GPU korelacja wywołań kolektywów, op/dtype/numel,
czas wejścia każdego ranku i związek z emitowanym tokenem oraz pomiar bez
profilera. Bez nich nie da się podać liczby kolektywów na krok/token ani
rozdzielić czasu transferu od oczekiwania. Skew sugeruje oczekiwanie;
down GEMV sugeruje możliwość ograniczenia liczby programów/tile overhead.
Żadna z tych hipotez nie jest jeszcze potwierdzonym zyskiem decode.

## Pierwszy eksperyment — tylko MoE down tile

- **Hipoteza:** przy C4 verify (do 20 wierszy) zmiana samego `BN` down GEMV
  z 8 na 64 zmniejszy liczbę programów dla N=4096 bez zmiany formatu wag,
  aktywacji, skal, `BK`, 4 warpów, routingu ani redukcji. Nie dotykamy gate,
  komunikacji, K4 ani arytmetyki; W4A8/MXFP4×FP8 to inny eksperyment.
- **Zmiana po uzyskaniu wyłącznego GPU:** dopuścić wartość 64 wyłącznie dla
  `R9K_GLM_PACKED_DOWN_BN` w strażniku `glm_moe.py` i `packed_gemv.py`;
  uruchomić BN8→BN64→BN8, a istniejącego `wide` nie używać w tym A–B–A.
  Wcześniejszy BN32 bez zysku podnosi próg dowodowy.
- **Poprawność:** przed pomiarem pełny routed MoE vs upstream na 5/20/32
  wierszach, nierówne trasy i skale, zmienne wejście przy replay grafu;
  następnie 4 równoległe zakończone odpowiedzi i NIAH strict/retrieval/
  format/completion/runtime, API stream/tool round-trip. Osobno mierzyć
  akceptację DFlash według pozycji.
- **Porównanie:** zamrożony protokół C1/C4, faktyczne 130816 tokenów wejścia,
  256 wyjścia, identyczne sampling/thinking/APC, jedna cold prime i trzy warm
  próby na leg; PCIe mapowanie/topologia/link idle i pod obciążeniem na każdym
  GPU; per-request decode, TTFT/prefill i aggregate osobno. Nie utożsamiać
  czasu kernela z czasem kroku lub decode end-to-end. Docelowe 261120 C1
  sprawdzać dopiero po dodatnim wyniku 128K. Protokół dla tego wariantu
  wymaga aktualizacji: obecny zamrożony `glm-packed-wide-vllm771-20261015`
  dotyczy **dwóch** zmian tile i nie może służyć do porównania BN64.
- **Odrzucenie/rollback:** odrzucić przy różnicy numerycznej, błędzie API,
  spadku jakości/akceptacji, OOM, preemption, braku prawdziwego C4 overlap
  albo gdy zysk decode nie przekroczy rozrzutu obu legów BN8. Powrót:
  `R9K_GLM_PACKED_DOWN_BN=8` i sprawdzony K4; nie zmieniać profilu
  produkcyjnego ani nie promować na podstawie mikrotestu.

Test komunikacji jest **przygotowany, nie uruchomiony**:
`profiling/glm_tp8_collectives.py` porównuje grafowe RCCL BF16 all-reduce
dla kandydackich 8/40/160 KiB na ośmiu procesach z kontrolą poprawności.
All-gather jest parametryzowany, bo jego rzeczywisty numel/dtype nie występuje
w obecnym śladzie; podać go dopiero po pomiarze host-op. Ten test wymaga
wyłącznego GPU i obowiązkowego PCIe preflightu. Nie przenosić wyników
TP2/TP4 na TP8.

## Wynik pierwszego eksperymentu: BN8 → BN64 → BN8

W izolowanym teście numerycznym pełny routed MoE z BN64 przeszedł 3/3
kształty (5/20/32 wiersze), zmienione trasy i wejścia pod replay grafu.
Następnie każda z trzech nóg uruchomiła ten sam obraz vLLM `77154ca6`,
TP8, DFlash K4, cztery sloty, 4,125 GiB KV/GPU i 210 W/GPU. Jedyną zmianą
było `R9K_GLM_PACKED_DOWN_BN=8` lub `64`; gate/up, komunikacja i reszta
profilu pozostały stałe. Każda noga miała osobny cold prime i trzy warm
powtórzenia identycznych promptów po 130816 tokenów, po 256 wymuszonych
tokenów wyjścia, przy temperaturze 1,0, top-p 0,95, seed 1234 i effort high.
Warm APC miał trafienia; C4 rzeczywiście osiągał cztery równoczesne decode.
Wszystkie nogi przeszły preflight i kontrolę połączeń PCIe pod obciążeniem
na ośmiu GPU, bez preemption i błędów runtime. NIAH z naturalnym EOS
zakończył się 4/4 strict/retrieval/format/completion w każdej nodze.

| Noga | C1 median decode/request | C4 median decode/request | C4 median aggregate incl. TTFT | C4 draft acceptance | C4 median TTFT |
|---|---:|---:|---:|---:|---:|
| BN8 pierwsza | 50,20 tok/s | 22,58 tok/s | 78,05 tok/s | 32,8% | 1381 ms |
| BN64 | 56,05 tok/s | 26,57 tok/s | 89,72 tok/s | 38,6% | 1381 ms |
| BN8 powtórka | 50,11 tok/s | 25,37 tok/s | 83,45 tok/s | 36,6% | 1350 ms |

BN64 ma wyższą medianę C4 niż obie nogi BN8, ale przewaga nad drugą
wynosi tylko 4,7% per-request i 7,5% aggregate, podczas gdy same nogi
BN8 różniły się o 12,4% i 6,9%. Akceptacja DFlash też się zmieniła.
**Ten test nie potwierdza zysku end-to-end ponad rozrzut baseline; BN64
nie promujemy.** C1 również ma nakładające się zakresy pojedynczych prób.
Wymuszone 256-tokenowe generacje służą do pomiaru prędkości, nie do oceny
jakości odpowiedzi. NIAH jest osobnym testem poprawności. Nadal nie mamy
znaczników czasu kroku z granicami draft/verify ani identyfikacji op/dtype/
numel każdego kolektywu, więc nie przypisujemy różnicy pojedynczemu kernelowi.

Pełny protokół, wszystkie próby, liczniki akceptacji według pozycji, pomiary
PCIe i logi leżą w ignorowanym katalogu
`bench/results/glm-down64-vllm771-20261015/`. Końcowy BN8/K4 pozostał
uruchomiony na porcie 8080. SGLang nie jest teraz uzasadniony konkretnym
operatorem eliminującym wykazany koszt.

## Aktualny ślad CPU+GPU C4 na vLLM 77154ca6

Osobny **diagnostyczny** przebieg z aktywnym profilerem CPU+GPU użył tego
samego obrazu i BN8/K4, 4 równoległych requestów, 130816 tokenów wejścia,
64 wymuszonych tokenów wyjścia, effort high, temperatury 1,0 i top-p 0,95.
Po cold prime warm APC osiągnął rzeczywisty overlap C4; wszystkie cztery
requesty zakończyły się poprawnie, bez preemption. Na wszystkich ośmiu GPU
zapisano idle i loaded PCIe. Surowy ślad oraz pełne próby znajdują się w
ignorowanym `bench/results/glm-c4-cpu-gpu-trace-retry-20261015/`. Skrypt
`profiling/glm_step_trace.py` łączy adnotacje CPU `execute_*` z kernelami GPU
przez identyfikatory korelacji HIP i paruje osiem ranków na wspólnej osi czasu.
Przebieg z profilerem **nie** jest porównywalnym pomiarem tok/s.

W czternastu powtarzalnych krokach `execute_20` (20 wierszy, cztery sesje)
mediana obwiedni GPU od pierwszego do ostatniego kernela dowolnego ranku
wynosiła **95,64 ms** (zakres 95,44–95,98 ms). Na rank medianowy przypadało
**39,34 ms sumy kerneli `_gemv`** i **8,41 ms sumy kerneli NCCL**. W jednym
reprezentatywnym kroku 84 `_gemv` ułożyły się w 42 pary: pierwsze kernele
par trwały łącznie 24,61 ms (mediana 0,584 ms), drugie 14,71 ms
(mediana 0,351 ms). Ich kolejność odpowiada gate/up i down w
`packed_gemv.py`, lecz sam ślad nazywa oba `_gemv`; podział jest wnioskiem
z kolejności wywołań. W tym samym kroku było 91 kerneli NCCL, 11 sparse
attention prefill ragged (łącznie 7,90 ms) i 11 paged logits (5,33 ms).
Mediana rozjazdu wejścia ośmiu ranków w sparowane epizody NCCL wynosiła
21,79 µs, a wyjścia 0,92 µs. To przemawia za częścią czasu spędzoną na
oczekiwaniu na późniejsze ranki; nie identyfikuje op/dtype/numel kolektywów.

Te sumy nie są udziałami ścieżki krytycznej ani prognozą zysku. Ślad
pokazuje jednak, że w rzeczywistym 20-wierszowym kroku C4 MoE ma znacznie
większy koszt niż NCCL, a samo zwiększenie `BN` down nie wystarczyło do
udowodnienia zysku end-to-end. Następny wariant powinien izolować koszt
gate/up albo liczbę oddzielnych przebiegów MoE; każde twierdzenie o
przyspieszeniu wymaga nowego zamrożonego A–B–A **bez profilera**.

## Bramka odczytu skal gate/up: test A–B–A

Wariant opt-in `R9K_GLM_PACKED_GATE_SCALE_GROUP=1` pobiera jedną skalę
E8M0 na 16 spakowanych bajtów zamiast ponawiać jej odczyt dla każdego
bajtu. Format wag, aktywacji, skal, kafle BN8, routing i K4 pozostały bez
zmian. Jedno-GPU mikrotest w grafie przeszedł tolerancję liczbową;
przy 20 wierszach mediana samego gate/up GEMV wynosiła 0,512 ms wobec
0,557 ms w obu sąsiednich pomiarach bazowych. Wyniki nie były bitowo
identyczne: przy 20 wierszach maksimum różnicy BF16 wyniosło 0,125,
a dokładnie równych było 99,993% elementów. Pełny routed MoE przeciw
upstream emulation przeszedł 3/3 prób dla 5/20/32 wierszy, zmiennych tras
i replay grafu. Mikrotest nie jest pomiarem modelu.

Pełny eksperyment użył tego samego obrazu vLLM `77154ca6`, 8× R9700,
TP8, FP8 KV, DFlash K4 i 130816 tokenów wejścia. Każda noga miała
osobny cold prime oraz **siedem** ciepłych powtórzeń C1 i C4 po 256
wymuszonych tokenów, temperatura 1,0/top-p 0,95/effort high. Trzy nogi
przeszły preflight PCIe dla ośmiu GPU, loaded-link, pełny overlap C4,
zero preemption, NIAH 4/4 i API contract ze streamingiem/tool callingiem.
Wyniki to mediany z median prędkości poszczególnych żądań w rundzie;
aggregate obejmuje TTFT, więc jest oddzielną wielkością.

| Noga | C1 decode/request | C4 decode/request | C4 aggregate incl. TTFT | C4 akceptacja draft | C4 warm TTFT |
|---|---:|---:|---:|---:|---:|
| Baza A | 49,26 tok/s | 24,50 tok/s | 78,53 tok/s | 34,7% | 1367 ms |
| Grupowane skale gate/up | 53,40 tok/s | 26,15 tok/s | 85,33 tok/s | 37,1% | 1510 ms |
| Baza C | 54,51 tok/s | 23,85 tok/s | 80,33 tok/s | 33,2% | 1381 ms |

Wariant B poprawił C4, ale C1 jest **wolniejszy od bazy C**, a akceptacja
DFlash różni się między nogami i pojedynczymi próbami. C1 miał zakresy
42,20–56,87 (A), 49,75–63,99 (B) i 42,49–57,98 tok/s (C).
**Nie ma potwierdzonego zysku end-to-end C1+C4 według zamrożonej reguły;
flaga pozostaje wyłączona i nie testujemy jej jako finalisty 256K.**
Pomocnicze C1 `sum(update_gaps_ms)/draft_rounds` miało mediany 45,43,
44,94 i 45,36 ms, ale nie jest to zmierzony czas kroku GPU ani pełny
decode i nie zmienia decyzji. Surowe próby, protokoły, preflighty, logi
i mikrotesty są w ignorowanych `bench/results/glm-gate-scale-*20261015/`.
W trakcie próby poprawiono fałszywy alarm preflightu: port podrzędny
reklamujący x16 jest prawidłowo ograniczany przez fizyczny port nadrzędny
x4; test regresji potwierdza też wykrywanie faktycznego spadku do x2.

## Kafel odczytu FP8 indexera: TILE64 → TILE256 → TILE64 (2026-10-17)

Ślad C4 wyżej zawierał 11 uruchomień paged logits na krok, łącznie 5,33 ms
na analizowanym ranku. Zbadaliśmy wyłącznie wielkość kafla tego czytnika,
bez zmiany wag, formatu FP8, arytmetyki, DFlash K4 ani pozostałych kerneli.
Flaga `R9K_GLM_PAGED_LOGITS_TILE` jest opt-in; domyślna wartość nadal wynosi
64. Wstępny TILE32 nie był bitowo zgodny dla wszystkich logitów FP32 (maks.
różnica około 1,9e-6), więc nie jest dopuszczony w kodzie produkcyjnym.

Izolowany grafowy mikrotest na `gfx1201` porównał pełne logity i top-k
bitowo z TILE64 dla 5 wierszy C1 i 20 wierszy C4, przy 128K/258048
tokenach kontekstu i pojemności modelu 262144. TILE128 i TILE256 przeszły
kontrolę. Przy C4/128K mediana samego czytnika wyniosła 0,535–0,525 ms
dla TILE64 i 0,347 ms dla TILE256. To nie jest oszczędność czasu kroku
ani całej odpowiedzi. Surowe próby, w tym odrzucony pilot TILE32, są w
ignorowanym `bench/results/glm-paged-logits-tiles-20261017/`.

Pełne A–B–A użyło tego samego obrazu `r9700/vllm:glm53-lmcache055-lru`
(vLLM `e97573215`), checkpointu Quark MXFP4, TP8, czterech slotów,
DFlash K4, FP8 target/draft KV i cache 128 GiB RAM / 512 GiB filesystem.
Protokół `bench/baseline-earkita/glm-paged-logits-128k.json` zamroził
identyczne prompty po 130816 tokenów, high, temperaturę 1,0, top-p 0,95,
seed 1234, jeden cold prime, jedną warmup i trzy mierzone partie C1/C4 po
256 wymuszonych tokenów. Osobny NIAH używał naturalnego EOS. Każda noga
miała odrębny namespace cache, potwierdzone trafienia L1, rzeczywisty
overlap C4, brak preemption i obowiązkowy zapis PCIe wszystkich ośmiu GPU
przed próbą i pod obciążeniem. Hash 116 źródeł, korpus, sampling i profil
K4 były identyczne; w tożsamości modelu różniła się tylko flaga kafla.

| Noga | C1 decode/request | C4 decode/request | C4 ITL p50 / p95 | C4 TTFT mediana |
|---|---:|---:|---:|---:|
| TILE64 A | 48,68 tok/s | 22,91 tok/s | 94,83 / 99,72 ms | 3105 ms |
| TILE256 B | 50,39 tok/s | 24,17 tok/s | 93,09 / 98,18 ms | 3139 ms |
| TILE64 C | 43,18 tok/s | 26,48 tok/s | 94,66 / 100,03 ms | 3425 ms |

Decode to mediana żądań z trzech partii; C1 ma n=3, C4 n=12. Wymuszony
256-tokenowy ciąg po krótkiej odpowiedzi NIAH nie miał zachowanego tekstu
strumienia i może zawierać generację po znaczniku końca tury. Zgodnie z
`testing.md` ten pomiar jest **diagnostyką syntetyczną**, nie miarą
rzeczywistego zadania programistycznego. Wniosek o przyspieszeniu wymaga
odrębnego, zamrożonego długiego zadania z EOS i zapisanym pełnym tekstem,
uruchomionego dla obu wariantów. Wszystkie trzy nogi przeszły NIAH 5/5
strict/retrieval/format/completion, bez błędów
runtime. Przepustowość aggregate C4, która **obejmuje TTFT**, miała mediany
72,18 / 72,50 / 73,78 tok/s odpowiednio A/B/C; nie jest to prędkość sesji.
Mediany akceptacji K4 C4 wynosiły 35,8% / 34,9% / 40,3%, a zakresy między
partiami A 31,2–37,3%, B 30,8–40,7%, C 37,5–42,0%. TILE256 skrócił medianę
odstępu strumienia C4 o około 1,6 ms względem obu prób TILE64, ale jego
mediana decode C4 była niższa od powtórzonej bazy C. **Nie ma potwierdzonego, powtarzalnego
zysku end-to-end; domyślne TILE64 pozostaje bez zmian.** Nie wnioskujemy z
tych danych o 256K, cold prefill ani zadaniach wieloetapowych z narzędziami.

Pełne protokoły, każdy request, przyjęcia według pozycji, telemetria PCIe,
logi i wyniki jakości są w `bench/results/glm-paged-tile64-a-20261017/`,
`bench/results/glm-paged-tile256-b-20261017/` i
`bench/results/glm-paged-tile64-c-20261017/` (ignorowane przez Git).
Oryginalne kontenery K4/LMCache zostały przywrócone po eksperymencie.

## Naturalny EOS i gęste FP8: K4 → opt-in → K4 (2026-10-17)

Żeby uniknąć mierzenia generacji po końcu odpowiedzi, zamrożony test
`bench/glm_completed_code.py` użył archiwum kodu o **131072 rzeczywistych
tokenach wejścia** i zadania `dependency_waves`. W każdej nodze najpierw
wykonano jedną cold rozgrzewkę, a potem trzy mierzone próby z naturalnym
EOS. Zachowano pełny tekst, reasoning, zdarzenia SSE, liczniki tokenów oraz
wynik niezależnego wykonania 169 przypadków. Prędkość decode jest liczona
z przyrostu rzeczywistego licznika tokenów między pierwszą i ostatnią
tekstową aktualizacją strumienia; TTFT i pierwszy burst są wyłączone.
Odstępy aktualizacji SSE nie są per-token ITL przy spekulacji.

Przez całe A–B–A działały ten sam obraz i checkpoint, TP8, DFlash K4, FP8
KV, LMCache, argumenty serwera, prompt, temperatura 1,0, top-p 0,95,
seed 1234 i effort high. Jedyną różnicą środowiska modelu była flaga
`R9K_GLM_DENSE_FP8` (0/1/0); osobne sole cache izolowały cold prime.
Każda ciepła próba miała **130176** tokenów trafienia GPU prefix cache.
Mapowanie HIP/BDF, topologia i linki pod obciążeniem ośmiu GPU przeszły
kontrolę w każdej nodze. Wszystkie **9/9 mierzone odpowiedzi** zakończyły
się poprawnym kodem (169/169 przypadków na odpowiedź). Nie wykonywano
równolegle innych testów GPU.

| Noga | C1 decode mediana (zakres) | TTFT mediana | Cała odpowiedź mediana | Akceptacja draftu mediana |
|---|---:|---:|---:|---:|
| K4 A | 61,41 (57,80–68,21) tok/s | 987 ms | 14,79 s | 52,4% |
| Gęste FP8 | 59,77 (56,73–63,33) tok/s | 957 ms | 14,45 s | 50,1% |
| K4 C | 63,87 (63,36–67,11) tok/s | 962 ms | 8,88 s | 57,1% |

Wartości całego czasu odpowiedzi nie są bezpośrednim zyskiem, bo ukończone
odpowiedzi miały różne długości: mediany 797 / 765 / 503 tokenów. Podobnie
akceptacja DFlash zmieniała się między próbkami; zapisano ją osobno, bez
przypisywania jej zmian do kernela. Mediany odstępu aktualizacji SSE wyniosły
50,54 / 50,39 / 50,56 ms. Izolowany kernel gęstego FP8 przeszedł 18/18
przypadków GPU i był szybszy dla odpowiednich kształtów, lecz **pełny model
nie wykazał powtarzalnego zysku decode**. Flaga pozostaje wyłączona. To
test C1 na jednym zadaniu programistycznym, nie kwalifikacja C4, 256K ani
wieloetapowego agenta z narzędziami.

Wszystkie próby, pełne odpowiedzi, raw Prometheus, licznik akceptacji,
protokół, kod orakla i PCIe są w ignorowanym
`bench/results/glm-dense-code-128k-20261017/`. K4 został przywrócony;
kontener kandydata jest zatrzymany. Następny eksperyment powinien dotyczyć
większego składnika ścieżki krytycznej, zwłaszcza MoE albo verify, z tą
samą bramką zakończonej odpowiedzi i jakości narzędzi.

## Sparse verify: izolowany test 4/8 warpów (2026-10-17)

W śladzie C4 sparse verify zużył 7,90 ms sumy kerneli na analizowanym
ranku. Obecny czytnik FP8 używa 8 warpów dla 1–8 zapytań, lecz 4 dla
20-wierszowego verify C4. `tuning/tune_glm_sparse_verify_warps.py`
sprawdził ten sam kernel przy 5 i 20 wierszach, 16 głowach, latentnym
wymiarze 512 i 2048 wybranych pozycjach KV. Izolowana karta gfx1201
przeszła kontrolę PCIe przed próbą i pod obciążeniem. Wyniki dla 4 i 8
warpów były bitowo identyczne na tych danych; osiem grafowych powtórzeń
dało mediany 1,574 i 0,393 ms na kernel przy 20 wierszach (przy pięciu:
1,576 i 0,396 ms). Surowe wyniki są w ignorowanym
`bench/results/glm-sparse-verify-warps-20261017/`.

To mocna hipoteza o koszcie spillowania/redukcji w wariancie 4-warpowym,
ale **nie dowód przyspieszenia kroku ani całego decode**. Kolejna bramka
wymaga opt-in ograniczonego do GLM/gfx1201/FP8 i 9–32 wierszy, pełnego
testu numerycznego z nierównymi długościami i grafowym replayem oraz
porównania K4→kandydat→K4 na zakończonych zadaniach C4. Działającego
profilu K4 nie zmieniono.
