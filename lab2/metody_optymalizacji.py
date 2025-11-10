import time, math

def odleglosc_kwadrat(p1, p2):
    """Oblicza kwadrat odległości euklidesowej między dwoma punktami."""
    return sum((c1 - c2) ** 2 for c1, c2 in zip(p1, p2))


def p2_dominuje_p1(p2, p1, kierunki):
    """
    Sprawdza, czy punkt p2 dominuje punkt p1, biorąc pod uwagę kierunki optymalizacji.
    Dominacja zachodzi, gdy p2 jest lepszy lub równy we wszystkich kryteriach
    i ściśle lepszy w co najmniej jednym.

    Argumenty:
        p2, p1 : list/tuple liczb
            Porównywane punkty.
        kierunki : list/tuple liczb (1 lub -1)
            1 = minimalizacja, -1 = maksymalizacja.
    
    Zwraca:
        (bool, int): para (czy_dominuje, liczba_porównań_współrzędnych)
    """
    # Przeskalowanie punktów tak, aby wszystkie kryteria traktować jako minimalizację
    p1_scaled = [x1 * k for x1, k in zip(p1, kierunki)]
    p2_scaled = [x2 * k for x2, k in zip(p2, kierunki)]

    liczba_porownan_wsp = 0
    jest_lepszy_lub_rowny = True
    jest_scisle_lepszy = False

    # Jedno przejście przez współrzędne – zliczamy porównania
    for x1, x2 in zip(p1_scaled, p2_scaled):
        liczba_porownan_wsp += 1  # porównania
        if x2 > x1:  # gorszy w którejś współrzędnej => nie dominuje
            jest_lepszy_lub_rowny = False
            break
        if x2 < x1:
            jest_scisle_lepszy = True

    return jest_lepszy_lub_rowny and jest_scisle_lepszy, liczba_porownan_wsp


def znajdz_front_pareto(X, kierunki):
    """
    Znajduje zbiór punktów niezdominowanych (front Pareto) z uwzględnieniem
    kierunków optymalizacji i zlicza osobno:
        - liczbę porównań punktów,
        - liczbę porównań współrzędnych,
        - całkowity czas obliczeń.

    Argumenty:
        X (list of list/tuple): Lista punktów wejściowych.
        kierunki (list/tuple): Lista z wartościami 1 (minimalizacja) lub -1 (maksymalizacja)
                               dla każdego kryterium.

    Zwraca:
        tuple:
            - lista P punktów niezdominowanych,
            - liczba_porownan_punktow,
            - liczba_porownan_wspolrzednych,
            - czas_obliczen (sekundy)
    """
    start_time = time.perf_counter()

    liczba_porownan_punktow = 0
    liczba_porownan_wspolrzednych = 0
    P = []

    for p1 in X:
        jest_zdominowany = False
        for p2 in X:
            if p1 is p2:
                continue

            liczba_porownan_punktow += 1
            dominuje, porownania_wsp = p2_dominuje_p1(p2, p1, kierunki)
            liczba_porownan_wspolrzednych += porownania_wsp

            if dominuje:
                jest_zdominowany = True
                break

        if not jest_zdominowany:
            P.append(p1)

    P_unikalne = list(dict.fromkeys(map(tuple, P)))
    czas_obliczen = time.perf_counter() - start_time

    return P_unikalne, liczba_porownan_punktow, liczba_porownan_wspolrzednych, czas_obliczen


def znajdz_front_z_filtracja(X, kierunki):
    """
    Znajduje front Pareto z agresywną filtracją, zliczając:
      - liczbę porównań punktów,
      - liczbę porównań współrzędnych,
      - całkowity czas działania.

    Argumenty:
        X (list of list/tuple): Lista punktów wejściowych.
        kierunki (list/tuple): Lista z wartościami 1 (minimalizacja) lub -1 (maksymalizacja)
                               dla każdego kryterium.

    Zwraca:
        tuple:
            - lista P punktów niezdominowanych,
            - liczba_porownan_punktow,
            - liczba_porownan_wspolrzednych,
            - czas_obliczen (sekundy)
    """
    start_time = time.perf_counter()

    liczba_porownan_punktow = 0
    liczba_porownan_wspolrzednych = 0
    P = []  # Zbiór punktów niezdominowanych
    X_do_przegladu = list(X)  # Kopia zbioru do modyfikacji

    while X_do_przegladu:
        kandydat = X_do_przegladu[0]

        # --- Szukaj lepszego kandydata ---
        for inny_punkt in X_do_przegladu[1:]:
            liczba_porownan_punktow += 1
            dominuje, por_wsp = p2_dominuje_p1(inny_punkt, kandydat, kierunki)
            liczba_porownan_wspolrzednych += por_wsp

            if dominuje:
                kandydat = inny_punkt

        P.append(kandydat)

        # --- Filtracja zdominowanych punktów ---
        liczba_porownan_punktow += len(X_do_przegladu)
        nowa_lista = []
        for p in X_do_przegladu:
            if p == kandydat:
                continue
            dominuje, por_wsp = p2_dominuje_p1(kandydat, p, kierunki)
            liczba_porownan_wspolrzednych += por_wsp
            if not dominuje:
                nowa_lista.append(p)
        X_do_przegladu = nowa_lista

    czas_obliczen = time.perf_counter() - start_time
    return P, liczba_porownan_punktow, liczba_porownan_wspolrzednych, czas_obliczen


def algorytm_punkt_idealny(X, kierunki):
    """
    Algorytm znajdowania frontu Pareto oparty o punkt idealny,
    zliczający porównania punktów, współrzędnych oraz czas wykonania.

    Argumenty:
        X (list of list/tuple): Lista punktów wejściowych.
        kierunki (list/tuple): 1 = minimalizacja, -1 = maksymalizacja.

    Zwraca:
        tuple:
            - lista P punktów niezdominowanych,
            - liczba_porownan_punktow,
            - liczba_porownan_wspolrzednych,
            - czas_obliczen (sekundy)
    """
    if not X:
        return [], 0, 0, 0.0

    start_time = time.perf_counter()

    liczba_porownan_punktow = 0
    liczba_porownan_wspolrzednych = 0
    P = []

    # --- Krok 2–3: Znajdź punkt idealny ---
    X_scaled = [[x * k for x, k in zip(p, kierunki)] for p in X]
    liczba_wymiarow = len(X[0])
    punkt_idealny = tuple(
    min(p[i] for p in X) if kierunki[i] == 1 else max(p[i] for p in X)
    for i in range(len(X[0]))
)

    # --- Krok 4–6: Oblicz odległości do punktu idealnego i posortuj ---
    odleglosci_posortowane = sorted(
        [(odleglosc_kwadrat(punkt_idealny, p_scaled), p_original)
         for p_scaled, p_original in zip(X_scaled, X)]
    )

    # --- Krok 7–9: Filtracja zdominowanych punktów ---
    X_do_przegladu = list(X)

    for dist, punkt_sprawdzany in odleglosci_posortowane:
        # Jeśli punkt został już usunięty, pomiń go
        if punkt_sprawdzany not in X_do_przegladu:
            continue

        # Punkt nie jest zdominowany — trafia na front Pareto
        P.append(punkt_sprawdzany)

        # Sprawdzenie i usuwanie punktów zdominowanych przez ten punkt
        liczba_porownan_punktow += len(X_do_przegladu)

        nowa_lista = []
        for p in X_do_przegladu:
            if p == punkt_sprawdzany:
                continue
            dominuje, por_wsp = p2_dominuje_p1(punkt_sprawdzany, p, kierunki)
            liczba_porownan_wspolrzednych += por_wsp
            if not dominuje:
                nowa_lista.append(p)
        X_do_przegladu = nowa_lista

    czas_obliczen = time.perf_counter() - start_time
    return P, liczba_porownan_punktow, liczba_porownan_wspolrzednych, czas_obliczen, punkt_idealny

'''
#TEST
punkty_X = [
    (5,5), (3,6), (4,4), (5,3), (3,3),
    (1,8), (3,4), (4,5), (3,10), (6,6), (4, 1), (3, 5)
]
kierunki = [1, 1]

front_pareto, liczba_porownan_punktow, liczba_porownan_wspolrzednych, czas_obliczen = znajdz_front_pareto(punkty_X, kierunki)
print("Zbiór punktów wejściowych X:")
print(punkty_X)
print("\n------------------------------------\n")
print("Znaleziony zbiór punktów niezdominowanych P(X):")
print(front_pareto)
print(f"\nCałkowita liczba wykonanych porównań punktów: {liczba_porownan_punktow}")
print(f"\nCałkowita liczba wykonanych porównań współrzędnych: {liczba_porownan_wspolrzednych}")
print(f"Czas obliczeń: {czas_obliczen:.6f} s")

front_pareto, liczba_porownan_punktow, liczba_porownan_wspolrzednych, czas_obliczen = znajdz_front_z_filtracja(punkty_X, kierunki)
print("Zbiór punktów wejściowych X:")
print(punkty_X)
print("\n------------------------------------\n")
print("Znaleziony zbiór punktów niezdominowanych P(X):")
print(front_pareto)
print(f"\nCałkowita liczba wykonanych porównań punktów: {liczba_porownan_punktow}")
print(f"\nCałkowita liczba wykonanych porównań współrzędnych: {liczba_porownan_wspolrzednych}")
print(f"Czas obliczeń: {czas_obliczen:.6f} s")

front_pareto, liczba_porownan_punktow, liczba_porownan_wspolrzednych, czas_obliczen = algorytm_punkt_idealny(punkty_X, kierunki)
print("Zbiór punktów wejściowych X:")
print(punkty_X)
print("\n------------------------------------\n")
print("Znaleziony zbiór punktów niezdominowanych P(X):")
print(front_pareto)
print(f"\nCałkowita liczba wykonanych porównań punktów: {liczba_porownan_punktow}")
print(f"\nCałkowita liczba wykonanych porównań współrzędnych: {liczba_porownan_wspolrzednych}")
print(f"Czas obliczeń: {czas_obliczen:.6f} s")
'''