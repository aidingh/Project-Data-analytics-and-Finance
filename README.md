# Project-Data-analytics-and-Finance
Project Data analytics and Finance, by Aidin Ghassemloi

Genom att ha ambition att starta eget och ett stort intresse för statistik, så börjades ett projekt som innefattar data och investerings analyser. Målet med projektet är att implementera ett analys verktyg som kan jag själv kan dra nytta av om jag skulle starta eget. Där jag kan utföra analyser och rapporter för att kunna investera åt andra eller för mig själv. Projektet är uppdelat i fyra kategorier och dess uppbyggnad är att förstå det teoretiska och sedan utföra en implementation av dessa. Den första kategorien är att analysera aktier och portföljer genom att implementera modeller som används inom finans. Genom att implementera olika metoder av att erhålla historiska avkastningar och risker av aktier och portföljer så har modeller implementerats. Den andra kategorin innefattar forecasting med hjälp av machine learning. Forecastings modellerna är till för att kunna förutse aktie priser, slutpriser eller andra variabler som man vill förutse, eller att undersöka en viss marknad som kan vara kopplad till aktien som man undersöker. Utöver denna kategori så har det även implementeras forecasting med neurala nätverk. Jag har tidigare arbetat med neurala nätverk och ville ha ytterliggare modeller inom mitt finanas verktyg. Slutligen så har jag implementerat handelsstrategier så som moving averages och plotta köp och sälj signaler på den historiska datan. De modeller och funktioner som är implementerade har förmågan att plottas grafiskt.

För ytterliggare information läs min github wiki. Där beskrivs vilka classer som har vilka modeller och funktioner!

Nedanför beskrivs modellerna mer i detalj och vad deras syfte är i mitt projekt.

{Finans modeller} För att kunna implementera modeller och utöka analymetoder så jag implementerat två sätt att erhålla den historiska avkastningen, vilket är simpel avkastning och logaritmisk avkastning. Avkastningarna går att erhålla på en avskild aktie, portfölj eller flera index.

Riskanalyser av investeringar: Med avkastning kommer risk. Därför har jag även implementerat riskanalyser av enskila aktier, portföljer och av olika index. Där olika varibler så som standard avvikelse, varience, portfölj volatilitet, systematisk risk och icke-systemtisk risk undersöks. Inom denna sektion undersöks även covarience och correlation mellan aktier.

Efficient frontier och CAPM-modellen: Markowitz portfölj optimering är en model som har implementerats för att hitta de mest effektiva portföljerna beroende på dess vikter i portföljen i avsikt på avkastning och risk. Den mest effektiva portföljen hittas med capital asset line som är en tangent till efficient frontier. Denna linjära tangent fås av CAPM-modellen. Denna model är implementerad för att få CAL till efficient frontier, beta värde och Sharpe ratio.

Normalfördelningas Modellen: En model som har implementerats genom att jag uppskattar normalfördelningens egenskaper. Modellen kan ge en sannolikhet tex att en årlig avkastning kommer vara större än en visst procentvärde som anges som input. Eller vad sannolikheten är att en förlust skall ske som är större än ett visst procentvärde. Intervall kan också anges som input tex att en årlig avkastning kommer va mellan två procentvärden.

{Forecasting med Machine Learning} Linjär regression/multipel regression modelerna används för två syften i mitt projekt. Det första syftet är att undersöka marknader. Den första funktionen scatter plottar datan och dess regressionslinje. Marknader är dock mer komplexa än att bara undersöka två variabler, där jag också har implementerat multipel regression model för att kunna hantera fler variabler. Det andra syftet är att förutse aktie priser eller andra aktie parametrar med linjär regression. Där jag har en variabel i min funktion som kan sättas till hur många dagar framåt man vill förutse aktie parametrarna.

SVM(Support Vector Machine(Regressor)) modellen används för att förutse aktie priser eller andra variabler som har med aktie datan att göra. Men den används också för att jämföra confidence värde alltså R²-värdet med andra machine learning algoritmer, vilket i mitt fall är linjär regression och monte carlo simulation för att få en uppfattning om hur bra förutsägelserna är. Nästa steg i mitt projekt är att implementera cross-validering för att få en ännu bättre uppfattning om hur bra algoritmerna är.

Monte carlo simulation av aktier: Används i samma aspekt som mina andra machine learning algoritmer, men i detta fall nyttjas slumptal från sannolikhets fördelningar.

{Forecasting med neurala nätverk} Forecastingen sker med RNN(Recurrent Neural Network) och RNN-Long Short Term Memory, eftersom att aktie datan är i en tidserie, alltså tiden där datan mäts är i samma intervall. Så passas det bra att nyttja ett RNN för att förutse aktie priser eller andra parametrar som finns i aktie datan. Inom denna analys har jag utvärderat olika aktiverings funktioner, förlust funktioner, batch storlekar, epoker, neuron-antal och neuron-lager. För att få så bra resultat emot validerings datauppsättningen som möjligt. Utöver detta så har jag utvärderat olika metoder att undvika overfitting och underfitting, i mitt fall använder dropouts för att unvika overfitting. RNN-LSTM har hittils givit mig ganska bra resultat där jag plottar validering mot det som är förutsagt av modellen samt plottar förlust mot epok för att få en uppfattning om hur bra modellen egentligen är.

{Handelsstrategier} Inom denna sektion har jag implementerat två sätt där jag kan plotta aktiedata med valda intervall. Vanling linjeplott och candlestickplot, där jag också kan plotta mina EMAS på dessa plotts för att undersöka trender i aktien(Bearish och Bullish). Genom att implementera dessa själv så utökar jag funktionallitet som passar mig, där jag kan ändra värden på min short/long windows samt välja sample tiden på mina candles. Utöver detta så har jag undersökt köp och sälj signaler för mina EMA´s, där köp och sälj signaler plottas när tex short EMA passerar/korsar long EMA. Detta koncept har jag undersökt för att sedan implementera en algoritm för algoritmisk trading.

{Projektets olika stadier} Mitt projekt följer sprintar där jag skriver en planering för en vecka i förväg. Alla kod skrivs först i script format för att utvärdera att dem fungerar korrekt, sedan skrivs alla funktioner om till ett objekt orienterat format för att unvika hårdkodning av parametrar samt att kunna kalla på alla klasser från en och samma main. Alla modeller/analys verktyg är implementerade i Python 3.6 i ett objekt orienterat format.

Det som skall implementeras i projektet just nu är fler machine learning algoritmer för att utöka det vertyget ännu mer. De modeller som skall implementeras är Naive Bayes modellen, K-Means clustering, Desicion Tree samt cross-validering.

Dessutom skall algoritmisk trading undersökas, där jag skall förska implementera en trading bot med ramverk så som Zipline/Quantopian så jag kan utföra backtesing av min algoritmer.

{Ramverk för projektet} De ramverk som har används mest i projektet är, Pandas, scikit learn, Keras, Numpy, Matplotlib
