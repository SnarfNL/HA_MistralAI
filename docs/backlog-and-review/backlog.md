# Backlog HA_MistralAI

Bron voor de GitHub Issues met label `backlog`. Eén sectie per story; na het aanmaken
zijn **de issues leidend** en wordt dit bestand niet meer bijgewerkt (behalve bij een
nieuwe review). Achtergrond per story: [`docs/review/2026-09-review.md`](review/2026-09-review.md).

## Labels

| Label | Betekenis |
| --- | --- |
| `backlog` | Komt uit deze backlog |
| `type:bug` · `type:feature` · `type:improvement` · `type:tech` | Soort werk |
| `priority:must` · `priority:should` · `priority:could` · `priority:later` | MoSCoW |
| `release:1-stabiel` · `release:2-ha-native` · `release:3-slimmer` · `release:later` | Geplande release |
| `effort:S` (≤ 1 dag) · `effort:M` (2–4 dagen) · `effort:L` (≥ 1 week) | Menselijke inschatting |

## Instructie voor Claude Code

Maak voor elke `## MA-xx`-sectie hieronder één GitHub-issue met de `gh`-CLI:

- **Titel:** de kop zonder `## ` (bijv. `MA-02 · Web-search-antwoord in gespreksgeschiedenis`).
- **Labels:** uit de regel `Labels:`. Maak ontbrekende labels eerst aan.
- **Body:** alles onder de `Labels`/`Status`-regels tot de volgende `## MA-`-kop.
- **Status `klaar (PR #N)`:** issue aanmaken en direct sluiten met commentaar "Opgelost in #N".
- **Status `in review (PR #N)`:** issue open laten en commentaar "Wordt opgelost in #N" plaatsen.
- Controleer eerst of er al een issue met dezelfde `MA-xx`-titel bestaat; maak geen dubbele.
- Voeg alle issues toe aan het GitHub Project "HA_MistralAI roadmap" als dat bestaat.
- Rapporteer aan het eind een tabel `MA-xx → issuenummer`.

---

## MA-01 · TTS-stream robuust bij falende eerste zin

Labels: `backlog` `type:bug` `priority:must` `release:1-stabiel` `effort:S`
Status: in review (PR #39)

Als gebruiker van een voice satellite wil ik dat het antwoord hoorbaar blijft als één zin mislukt, zodat mijn speaker niet stil blijft of ruis afspeelt.

**Acceptatiecriteria**

- [ ] Faalt zin 0 met een 4xx of time-out, dan begint de stream nog steeds met een geldige RIFF/WAVE-header.
- [ ] De overige zinnen worden afgespeeld; de mislukte zin wordt gelogd op WARNING.
- [ ] Unit-test simuleert een fout in zin 0 en valideert de header van de output.

**Techniek:** elke zin geeft zijn header apart door; de stream gebruikt de header van de eerste zin die echt audio levert. Review-bevinding C1.

**Schatting:** Claude 45 min + test 20 min · ≈ 0,5–1,5 M tokens

---

## MA-02 · Web-search-antwoord in gespreksgeschiedenis

Labels: `backlog` `type:bug` `priority:must` `release:1-stabiel` `effort:S`
Status: open

Als gebruiker wil ik na een zoekvraag een vervolgvraag kunnen stellen (“en morgen?”), zodat het gesprek natuurlijk blijft.

**Acceptatiecriteria**

- [ ] Na een beantwoorde zoekvraag via trigger of `always` staat het antwoord als `AssistantContent` in de chat_log.
- [ ] Een vervolgvraag binnen dezelfde `conversation_id` krijgt het vorige antwoord als context.
- [ ] Het antwoord wordt naar TTS gestreamd zoals een normaal antwoord.

**Techniek:** `chat_log.async_add_assistant_content_without_tools(...)` of de reply als delta-stream aanbieden, daarna `async_get_result_from_chat_log`. Review-bevinding C2.

**Schatting:** Claude 30 min + test 15 min · ≈ 0,5–1 M tokens

---

## MA-03 · Time-outs overal correct afhandelen

Labels: `backlog` `type:bug` `priority:must` `release:1-stabiel` `effort:S`
Status: open

Als beheerder wil ik dat een trage of onbereikbare Mistral-API netjes wordt afgehandeld, zodat HA niet vol tracebacks loopt en de integratie zichzelf herstelt.

**Acceptatiecriteria**

- [ ] Time-out bij setup geeft `ConfigEntryNotReady` (automatische retry).
- [ ] Time-out in de config flow geeft de fout `cannot_connect`, niet `unknown`.
- [ ] Time-outs in STT, TTS, conversation en AI Task geven een nette fout zonder traceback.

**Techniek:** overal `(aiohttp.ClientError, TimeoutError)` afvangen; later centraliseren in de API-client (MA-10). Review-bevinding C6.

**Schatting:** Claude 30 min + test 10 min · ≈ 0,5–1 M tokens

---

## MA-04 · Eén herbruikbare web-search-agent zonder gelekte context

Labels: `backlog` `type:bug` `priority:must` `release:1-stabiel` `effort:M`
Status: open

Als beheerder wil ik dat de integratie niet bij elke herstart een nieuwe agent in mijn Mistral-account aanmaakt en geen huisgegevens in die agent opslaat.

**Acceptatiecriteria**

- [ ] Na 10 herstarts bestaat er precies één agent “HA Mistral Web Search” per API-key.
- [ ] De agent-instructies bevatten geen entity-lijst, datum of gebruikersprompt.
- [ ] Een modelwijziging werkt de bestaande agent bij in plaats van een nieuwe te maken.
- [ ] De mapping HA-gesprek → Mistral-gesprek is begrensd (bijv. max 50 items, verloopt na 5 min).

**Techniek:** agent opzoeken via `GET /v1/agents` op naam; `_ws_convs` als veld in `MistralRuntimeData` met TTL. Review-bevindingen C3 en C4.

**Schatting:** Claude 1 u + test 20 min · ≈ 1,5–3 M tokens

---

## MA-05 · Continue conversation doet wat de schakelaar zegt

Labels: `backlog` `type:bug` `priority:must` `release:1-stabiel` `effort:S`
Status: open

Als gebruiker wil ik dat de microfoon alleen open blijft als ik die optie aanzet en de assistent echt een vraag stelt.

**Acceptatiecriteria**

- [ ] Optie uit: `continue_conversation` is altijd `False`, ook als het antwoord op `?` eindigt.
- [ ] Optie aan: alleen doorluisteren als het antwoord eindigt op een vraagteken (`?`, `？`, `;`), niet bij een `?` midden in de tekst.
- [ ] README beschrijft dat HA core dit standaard al doet.

**Techniek:** resultaat van `async_get_result_from_chat_log` overschrijven; heuristiek van core (`chat_log.continue_conversation`) hergebruiken. Review-bevinding C5.

**Schatting:** Claude 15 min + test 15 min · ≈ 0,3–0,6 M tokens

---

## MA-06 · Correct language lists and cleaned-up strings

Labels: `backlog` `type:bug` `priority:must` `release:1-stabiel` `effort:S`
Status: in progress (branch `feat/ma-06-ma-13-languages-and-voice-refresh`)

As an administrator I only want to pick languages Mistral supports, so I do not build a pipeline that fails silently.

**Acceptance criteria**

- [ ] STT reports the 13 languages of Voxtral Transcribe 2 (en, zh, hi, es, ar, fr, pt, ru, de, ja, ko, it, nl).
- [ ] TTS reports the 9 languages of Voxtral TTS (en, fr, es, pt, it, nl, de, hi, ar).
- [ ] `strings.json` and the translations no longer mention `stt_language` or the voice "Nova".

**Notes:** language support is about the words the model knows; the preset voices speak all 9 languages with their own accent. Aligned with the [STT docs](https://docs.mistral.ai/studio/audio/speech_to_text) and [TTS docs](https://docs.mistral.ai/studio/audio/text_to_speech). Review finding C7.

**Estimate:** Claude 15 min + test 5 min · ≈ 0.3–0.6 M tokens

---

## MA-07 · Reauth, backoff en vertaalde gesproken fouten

Labels: `backlog` `type:improvement` `priority:must` `release:1-stabiel` `effort:M`
Status: open

Als gebruiker wil ik bij een fout een korte, begrijpelijke zin in mijn eigen taal horen, en als beheerder wil ik direct een reauth-melding bij een ingetrokken key.

**Acceptatiecriteria**

- [ ] Een 401 tijdens gebruik start de reauth-flow (`entry.async_start_reauth`).
- [ ] Een 429 wordt maximaal 2 keer opnieuw geprobeerd met backoff en respecteert `Retry-After`; daarna volgt een melding “Mistral is even overbelast”.
- [ ] Alle `HomeAssistantError`s gebruiken `translation_domain`/`translation_key` (en, nl, fr); ruwe API-bodies staan alleen in het log.
- [ ] De conversation-agent geeft een `IntentResponse`-fout terug in plaats van een exception, zodat de satellite de fout uitspreekt.

**Schatting:** Claude 1–1,5 u + test 30 min · ≈ 1,5–3 M tokens

---

## MA-08 · CI en release-hygiëne

Labels: `backlog` `type:tech` `priority:must` `release:1-stabiel` `effort:S`
Status: klaar (PR #38)

Als maintainer wil ik dat elke PR automatisch gevalideerd wordt en dat versies kloppen, zodat regressies zoals #36 niet ongemerkt live gaan.

**Acceptatiecriteria**

- [x] GitHub Actions draaien hassfest, HACS-validatie, ruff en pytest op elke PR.
- [x] `manifest.json`-versie is gelijk aan de release-tag (gecontroleerd in CI).
- [x] Release notes staan in `CHANGELOG.md`, niet in de README.
- [x] `quality_scale` is uit het manifest verwijderd.
- [x] Wijzigingen gaan via branches en PR’s (`CONTRIBUTING.md`).

---

## MA-09 · AI Task: strikte structured output

Labels: `backlog` `type:bug` `priority:must` `release:1-stabiel` `effort:S`
Status: open

Als beheerder wil ik dat `ai_task.generate_data` met een `structure` altijd een dict teruggeeft of duidelijk faalt, zodat mijn automatisering niet op een string crasht.

**Acceptatiecriteria**

- [ ] Ongeldige JSON geeft een `HomeAssistantError` met `translation_key: json_parse_error`.
- [ ] `strict: true` wordt meegestuurd als het schema dat toelaat; getest op minstens 2 modellen.
- [ ] Een afwijking van de gevraagde structuur wordt als fout gemeld, niet alleen als warning.

**Techniek:** review-bevinding C8. Open vraag: ondersteunen alle modellen `strict: true`?

**Schatting:** Claude 30 min + test 15 min · ≈ 0,5–1 M tokens

---

## MA-10 · Refactor naar HA-patronen

Labels: `backlog` `type:tech` `priority:should` `release:2-ha-native` `effort:M`
Status: open

Als maintainer wil ik één plek voor API-calls en gedeelde entity-logica, zodat een API-wijziging op één plek wordt opgelost.

**Acceptatiecriteria**

- [ ] `entry.runtime_data` (getypeerd via `type MistralConfigEntry = ConfigEntry[MistralRuntimeData]`) vervangt `hass.data`.
- [ ] `api.py` bevat een `MistralClient` met methoden voor chat-stream, agents, conversations, speech, transcriptions, voices en models, plus centrale foutmapping.
- [ ] `entity.py` bevat een basisklasse met `device_info` en toegang tot de client.
- [ ] `mypy --strict` draait zonder fouten; `PARALLEL_UPDATES` is per platform gezet.
- [ ] Gedrag blijft gelijk: bestaande tests slagen zonder aanpassing van verwachtingen.

**Schatting:** Claude 2–3 u + test 1 u · ≈ 3–6 M tokens

---

## MA-11 · Tests op echte HA-code

Labels: `backlog` `type:tech` `priority:should` `release:2-ha-native` `effort:M`
Status: open

Als maintainer wil ik tests die tegen echte HA-klassen draaien, zodat breuken door HA-releases (zoals probatio in 2026.9) in CI opvallen.

**Acceptatiecriteria**

- [ ] Tests draaien op `pytest-homeassistant-custom-component` in een matrix met de minimale en de nieuwste HA-versie.
- [ ] Config flow, reauth en options flow hebben 100% dekking (Bronze-regel).
- [ ] Totale dekking ≥ 90%; de API wordt gemockt met `aioclient_mock` en opgenomen echte SSE-fixtures.
- [ ] `tests/_ha_stubs.py` is verwijderd.

**Schatting:** Claude 2–4 u + test 15 min · ≈ 4–8 M tokens

---

## MA-12 · Config subentries: meerdere agents

Labels: `backlog` `type:feature` `priority:should` `release:2-ha-native` `effort:L`
Status: open

Als beheerder wil ik meerdere Mistral-agents met elk een eigen model, prompt en tools kunnen maken (bijv. een snelle “Woonkamer”-agent op Ministral 8B en een “Onderzoeker” op Mistral Large met web search), en een aparte AI Task-configuratie.

**Acceptatiecriteria**

- [ ] De hoofd-entry bevat alleen de API-key; subentry-types `conversation`, `ai_task_data`, `stt` en `tts` hebben elk hun eigen opties.
- [ ] Een migratie (VERSION 1 → 2) zet bestaande opties om naar één conversation- en één AI Task-subentry; entity-ID’s blijven gelijk.
- [ ] Meerdere API-keys zijn mogelijk (unique_id op basis van key-hash in plaats van `DOMAIN`).
- [ ] Opties zijn gegroepeerd in secties (Model, Tools, Web search) met een “aanbevolen instellingen”-toggle.

**Techniek:** volg het patroon van de core-integraties `anthropic` en `openai_conversation`. Afhankelijk van MA-10 en MA-11.

**Schatting:** Claude 4–6 u + test 2 u · ≈ 6–12 M tokens

---

## MA-13 · Voice list from the account, with refresh button

Labels: `backlog` `type:feature` `priority:should` `release:2-ha-native` `effort:S`
Status: in progress (branch `feat/ma-06-ma-13-languages-and-voice-refresh`)

As an administrator I want the voice picker in Settings → Voice assistants to list the voices of my Mistral account, including custom voices made in Mistral Studio, and to refresh that list when I create a new one.

The voice is chosen only in the assistant settings (or via the `voice` option of `tts.speak`); the integration options dialog has no voice field (see #75).

**Acceptance criteria**

- [ ] The picker shows only the account's voices: presets and custom voices, with readable labels for presets (`Paul – Angry (English)`), custom voices first.
- [ ] A "Refresh voices" button (config category) on the Mistral AI TTS device re-fetches the list without restarting Home Assistant.
- [ ] If a refresh fails, the last good list is kept.
- [ ] There is no static fallback list: the picker is empty until the first successful fetch.
- [ ] Tests: a custom voice returned by the API appears in the picker; a failed refresh keeps the previous list; the picker is empty before the first fetch; label and ordering rules.
- [ ] CHANGELOG.md has a line under `### Unreleased`.

**Notes:** the fetch (`_async_fetch_voices`, paginated) runs at startup and on button press. Preset slugs such as `en_paul_neutral` work as `voice_id` (verified with `tts.speak`). No free input and no 24h timer. Decision on not moving TTS to a Wyoming server: `docs/adr/0001-tts-stays-in-the-integration.md`.

**Estimate:** Claude 30 min + test 15 min · ≈ 0.5–1.5 M tokens

---

## MA-14 · Diagnostics, repairs en reconfigure

Labels: `backlog` `type:feature` `priority:should` `release:2-ha-native` `effort:M`
Status: open

Als beheerder wil ik bij problemen een diagnostics-bestand kunnen delen en een melding krijgen als mijn model door Mistral is uitgefaseerd.

**Acceptatiecriteria**

- [ ] `diagnostics.py` levert opties, HA-versie, integratieversie en laatste fouten, met de API-key geredacteerd.
- [ ] Een model dat niet meer in `/v1/models` staat, geeft een fixbare repair issue met een voorgesteld vervangend model.
- [ ] Een reconfigure flow laat de API-key wijzigen zonder de integratie te verwijderen.

**Schatting:** Claude 1,5–2 u + test 30 min · ≈ 2–4 M tokens

---

## MA-15 · Web search als external tool call

Labels: `backlog` `type:improvement` `priority:should` `release:2-ha-native` `effort:M`
Status: open

Als gebruiker wil ik dat zoekresultaten betrouwbaar en veilig in het antwoord terechtkomen, en als beheerder wil ik in de debug-trace zien dat er gezocht is.

**Acceptatiecriteria**

- [ ] Een zoekactie verschijnt in de chat_log als `ToolInput(external=True)` met bijbehorend `ToolResultContent`.
- [ ] Resultaten gaan als `tool`-bericht naar het model, niet als `user`-bericht.
- [ ] Meerdere zoekvragen in één beurt worden allemaal uitgevoerd (maximaal 3, parallel).
- [ ] Optioneel wordt de thuislocatie (stad en land uit HA) meegegeven voor lokale vragen.
- [ ] De modes `always` en trigger worden vervangen of als deprecated gemarkeerd, met een migratiepad.

**Schatting:** Claude 1,5–2 u + test 45 min · ≈ 2–4 M tokens

---

## MA-16 · Nederlandse zinssplitsing voor TTS

Labels: `backlog` `type:improvement` `priority:should` `release:2-ha-native` `effort:S`
Status: open

Als Nederlandstalige gebruiker wil ik dat de assistent niet halverwege “bijv.” of “o.a.” pauzeert, en dat lange zinnen zonder punt niet voor vertraging zorgen.

**Acceptatiecriteria**

- [ ] De afkortingenlijst bevat NL, FR en DE (bijv., o.a., d.w.z., enz., m.b.t., z.B., usw., p.ex.).
- [ ] Na 200 tekens zonder terminator wordt geflusht op `,`, `;` of `:`.
- [ ] Unit-tests met Nederlandse voorbeeldzinnen.

**Schatting:** Claude 30 min + test 15 min · ≈ 0,5–1 M tokens

---

## MA-17 · Tokengebruik en latency zichtbaar

Labels: `backlog` `type:feature` `priority:should` `release:2-ha-native` `effort:M`
Status: open

Als beheerder wil ik zien hoeveel tokens en tijd een gesprek kost, zodat ik het juiste model en het free tier kan kiezen.

**Acceptatiecriteria**

- [ ] Per beurt worden `input_tokens`, `output_tokens` en time-to-first-token via `chat_log.async_trace` vastgelegd, zichtbaar in de Assist-debug.
- [ ] Optionele (standaard uitgeschakelde) sensoren tonen tokens vandaag en requests vandaag per subentry.
- [ ] Het `usage`-object uit de laatste SSE-chunk wordt geparsed zonder de bestaande deltaverwerking te breken.

**Schatting:** Claude 1–1,5 u + test 30 min · ≈ 1,5–3 M tokens

---

## MA-18 · STT context biasing met huisnamen

Labels: `backlog` `type:feature` `priority:could` `release:3-slimmer` `effort:S`
Status: open

Als gebruiker wil ik dat “zet de Hue Go in de serre aan” goed verstaan wordt, zodat eigen apparaat- en ruimtenamen niet verbasterd worden.

**Acceptatiecriteria**

- [ ] Optie “Gebruik namen uit mijn huis” stuurt maximaal 100 `context_bias`-termen mee: exposed entity-namen, aliassen, ruimtes en verdiepingen.
- [ ] Selectie is deterministisch (aliassen en ruimtes eerst) en gecachet; bij registry-wijzigingen wordt de cache ververst.
- [ ] Meting op 20 Nederlandse testopnames: word error rate op namen daalt aantoonbaar, anders blijft de optie standaard uit.

**Risico:** Mistral noemt context biasing experimenteel buiten het Engels.

**Schatting:** Claude 45 min + test 1 u · ≈ 1–2 M tokens

---

## MA-19 · Camerabeelden in het gesprek

Labels: `backlog` `type:feature` `priority:could` `release:3-slimmer` `effort:M`
Status: open

Als gebruiker wil ik vragen “wie staat er bij de voordeur?” en een beschrijving van het camerabeeld horen.

**Acceptatiecriteria**

- [ ] De conversation-agent verwerkt `UserContent.attachments` (afbeeldingen) en stuurt ze als `image_url` naar een vision-capabel model.
- [ ] Een tool “bekijk camera” haalt een snapshot van een exposed camera op en voegt die toe aan de beurt.
- [ ] Kiest de beheerder een model zonder vision, dan geeft de config flow een waarschuwing.
- [ ] Snapshots worden niet opgeslagen buiten de beurt.

**Schatting:** Claude 1,5–2 u + test 30 min · ≈ 2–4 M tokens

---

## MA-20 · Persoonlijk geheugen

Labels: `backlog` `type:feature` `priority:could` `release:3-slimmer` `effort:M`
Status: open

Als gebruiker wil ik zeggen “onthoud dat ik op dinsdag thuiswerk”, zodat de assistent daar later rekening mee houdt.

**Acceptatiecriteria**

- [ ] LLM-tools `remember_fact`, `forget_fact` en `list_facts` slaan feiten op in HA `Store`, per HA-gebruiker (of gedeeld als er geen gebruiker bekend is).
- [ ] Opgeslagen feiten worden aan de systeemprompt toegevoegd (maximaal ongeveer 1.000 tokens).
- [ ] Een beheerder kan het geheugen inzien en wissen via een service-action.
- [ ] Standaard uit; bij aanzetten verschijnt uitleg over wat er wordt opgeslagen.

**Schatting:** Claude 1,5–2 u + test 45 min · ≈ 2–4 M tokens

---

## MA-21 · Home Brief-blueprint

Labels: `backlog` `type:feature` `priority:could` `release:3-slimmer` `effort:S`
Status: open

Als gebruiker wil ik elke avond een korte gesproken samenvatting van wat er in huis gebeurde (camera-events, deurbel, energie), zoals Gemini’s Home Brief.

**Acceptatiecriteria**

- [ ] Een blueprint verzamelt snapshots en events van de dag en roept `ai_task.generate_data` aan.
- [ ] Het resultaat gaat naar een notificatie en optioneel naar `tts.speak` op een gekozen speaker.
- [ ] README bevat installatie en voorbeeldoutput.

**Schatting:** Claude 30–45 min + test 30 min · ≈ 0,5–1,5 M tokens

---

## MA-22 · AI Task: afbeeldingen genereren

Labels: `backlog` `type:feature` `priority:could` `release:3-slimmer` `effort:M`
Status: open

Als beheerder wil ik `ai_task.generate_image` met Mistral kunnen gebruiken, bijvoorbeeld voor een dagelijkse dashboardafbeelding.

**Acceptatiecriteria**

- [ ] De entity ondersteunt `AITaskEntityFeature.GENERATE_IMAGE` via Mistrals `image_generation`-tool.
- [ ] Het resultaat bevat de bytes, het MIME-type en het model; fouten zijn vertaald.
- [ ] Werkt met attachments als referentie waar de API dat ondersteunt.

**Schatting:** Claude 1–1,5 u + test 20 min · ≈ 1,5–3 M tokens

---

## MA-23 · Dynamische modellijst

Labels: `backlog` `type:feature` `priority:could` `release:3-slimmer` `effort:S`
Status: open

Als beheerder wil ik nieuwe Mistral-modellen en mijn fine-tunes kunnen kiezen zonder op een integratie-update te wachten.

**Acceptatiecriteria**

- [ ] De modeldropdown combineert de aanbevolen lijst met `/v1/models` (gefilterd op chat-capabel).
- [ ] Aanbevolen modellen blijven bovenaan met hun label; vrije invoer is toegestaan.
- [ ] De capabilities uit `/v1/models` (vision, function calling) bepalen welke features beschikbaar zijn.

**Schatting:** Claude 30–45 min + test 15 min · ≈ 0,5–1,5 M tokens

---

## MA-24 · Realtime STT

Labels: `backlog` `type:feature` `priority:could` `release:3-slimmer` `effort:L`
Status: open

Als gebruiker wil ik dat de transcriptie al loopt terwijl ik praat, zodat het antwoord sneller komt.

**Acceptatiecriteria**

- [ ] Een STT-modus “realtime” streamt PCM naar `wss://api.mistral.ai/v1/audio/transcriptions/realtime` met `voxtral-mini-transcribe-realtime-2602`.
- [ ] De eindtranscriptie is binnen 300 ms na het einde van de spraak beschikbaar (gemeten in de pipeline-debug).
- [ ] Bij een WebSocket-fout valt de entity terug op batch-transcriptie.

**Schatting:** Claude 3–5 u + test 1 u · ≈ 5–10 M tokens

---

## MA-25 · Automatisering opstellen met spraak

Labels: `backlog` `type:feature` `priority:could` `release:3-slimmer` `effort:L`
Status: open

Als gebruiker wil ik zeggen “doe elke werkdag om 7 uur de keukenlamp aan” en een automatisering als concept krijgen.

**Acceptatiecriteria**

- [ ] Een LLM-tool genereert automatiserings-YAML en valideert die met HA’s eigen schema.
- [ ] Het concept verschijnt als melding of repair met knoppen “Opslaan” en “Negeren”; nooit automatisch actief.
- [ ] Alleen entities die aan Assist zijn exposed kunnen gebruikt worden.

**Schatting:** Claude 3–5 u + test 1 u · ≈ 5–10 M tokens

---

## MA-26 · Handleidingen bevragen

Labels: `backlog` `type:feature` `priority:could` `release:3-slimmer` `effort:L`
Status: open

Als gebruiker wil ik vragen “hoe ontkalk ik de koffiemachine?” en antwoord krijgen uit de handleiding die ik heb geüpload.

**Acceptatiecriteria**

- [ ] De beheerder koppelt een Mistral document library (ID) aan een conversation-subentry.
- [ ] Het model krijgt een tool die via de Agents API de library doorzoekt.
- [ ] Antwoorden noemen de bron (documentnaam) in de debug-trace.

**Schatting:** Claude 2–4 u + test 45 min · ≈ 3–6 M tokens

---

## MA-27 · Voice cloning vanuit HA

Labels: `backlog` `type:feature` `priority:could` `release:3-slimmer` `effort:M`
Status: open

Als beheerder wil ik vanuit HA een eigen stem aanmaken uit een audiobestand, zodat ik niet naar Mistral Studio hoef.

**Acceptatiecriteria**

- [ ] Een service-action `mistral_conversation.create_voice` accepteert een mediabestand, naam en taal.
- [ ] De action vereist een expliciete toestemmingsbevestiging, conform Mistrals gebruiksbeleid.
- [ ] De nieuwe stem verschijnt na afloop direct in de voice-dropdowns.

**Schatting:** Claude 1–1,5 u + test 20 min · ≈ 1,5–3 M tokens

---

## MA-28 · Barge-in en speech-to-speech

Labels: `backlog` `type:feature` `priority:later` `release:later` `effort:L`
Status: open

Als gebruiker wil ik de assistent kunnen onderbreken terwijl hij praat, zoals bij ChatGPT Voice en Gemini Live.

**Toelichting:** wacht tot de HA-voice-pipeline full-duplex ondersteunt. Tot die tijd alleen de architectuurdiscussies bij Home Assistant volgen.

---

## MA-29 · Core-inclusie in Home Assistant

Labels: `backlog` `type:tech` `priority:later` `release:later` `effort:L`
Status: open

Als maintainer wil ik op termijn kunnen kiezen voor opname in Home Assistant core.

**Toelichting:** pas zinvol na MA-08, MA-10 en MA-11. Vereist een losse PyPI-library, strict typing en minimaal Silver op de Quality Scale.
