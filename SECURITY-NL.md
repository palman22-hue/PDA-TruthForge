# Security Policy

## Supported versions

PDA-TruthForge is momenteel een onderzoeks-/prototypeproject en wordt alleen in de `main` branch actief onderhouden.  
Andere branches of forks worden niet ondersteund voor security-ondersteuning.

## Deployment model

PDA-TruthForge is ontworpen voor lokaal gebruik, bovenop een lokaal geïnstalleerde LLM-runtime (zoals Ollama met het `mistral`-model).  
De repository biedt **geen** publieke, door de maintainer beheerde API of online dienst; eventuele externe blootstelling (bijvoorbeeld via een zelfgebouwde webservice) valt onder verantwoordelijkheid van de deployer. 

## Data & privacy

- Invoerdata (prompts, context) worden lokaal verwerkt via de gekozen LLM-runtime.  
- Eventuele logs, configuratiebestanden en sessiegegevens worden lokaal opgeslagen op het systeem van de gebruiker.  
- De maintainer ontvangt geen automatische telemetry of gebruikersdata. 

Gebruikers zijn zelf verantwoordelijk voor:
- Het beschermen van gevoelige data die zij via PDA-TruthForge aan een model aanbieden.
- De beveiliging van hun eigen host-systeem, besturingssysteem en runtime (Ollama, Python-omgeving, etc.). 

## Reporting a vulnerability

Als je een mogelijke kwetsbaarheid of ernstig misbruikrisico in PDA-TruthForge ontdekt, verzoeken wij je gebruik te maken van **responsible disclosure**. 

Als Private Vulnerability Reporting is ingeschakeld voor deze repository:
1. Gebruik de knop **“Report a vulnerability”** op de GitHub‑pagina van de repository om een privérapport in te dienen. 

Als Private Vulnerability Reporting niet beschikbaar is:
1. Open een minimaal publiek issue met:
   - Alleen een korte beschrijving dat je een mogelijke kwetsbaarheid hebt gevonden.
   - Geen technische details of exploitcode.
2. Wacht op reactie van de maintainer voor een geschikt privé-kanaal om details uit te wisselen.

Vermeld, waar mogelijk:
- Beschrijving van de kwetsbaarheid of het misbruikscenario.
- Reproduceerbare stappen of proof-of-concept (alleen via privé-kanaal).  
- Relevante omgeving (OS, Python-versie, Ollama-versie, modelversie).

Er wordt gestreefd naar een eerste reactie binnen 14 dagen, inclusief een indicatie van de vervolgstappen. 

## Scope

In scope:
- De code in deze repository (`main` branch), inclusief:
  - Orchestratie- en architectuurcode (bijv. `api.py`, orchestratie-/routinglagen).
  - Modules voor geheugen, emoties, reflectie en gerelateerde AI-logica.
  - Installatie- en configuratiescripts die in de README worden genoemd.

Out of scope:
- Externe LLM-runtime (zoals Ollama, Mistral of andere modellen en hun eigen kwetsbaarheden).
- Door derden gebouwde services, wrappers of deploys die PDA-TruthForge aan internet blootstellen.
- Hardware- en OS-specifieke issues die niet direct uit de code van deze repository voortkomen.

## Responsible use

PDA-TruthForge is bedoeld als ethisch fundament voor LLM-systemen.  
Gebruikers worden nadrukkelijk aangemoedigd het systeem niet te gebruiken voor:

- Het ontwikkelen of ondersteunen van schadelijke, misleidende of onethische AI-toepassingen.
- Het omzeilen van veiligheidsmechanismen van bovenliggende modellen of platformen.

Indien je misbruik van PDA-TruthForge in het wild aantreft, kun je dit eveneens via bovenstaande route melden.

