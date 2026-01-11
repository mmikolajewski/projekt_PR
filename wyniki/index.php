<?php
//phpinfo();

function kernelC($t, $o, $N, $R, $k)
{
	global $blockDimX;
	global $blockDimY;

	global $threadIdxX;
	global $threadIdxY;

	global $blockIdxY;
	global $blockIdxX;

	global $gridDimX;
	global $gridDimY;

	global $IN;
	global $OUT;
	global $CACHE;

    $ii = 1;

	$outSize = $N - 2 * $R;
	$bs = $blockDimX;
	$tileW = $bs + 2 * $R;
	$tileH = $bs + 2 * $R;

	$tid = $threadIdxY * $blockDimX + $threadIdxX;
	$nThreads = $blockDimX * $blockDimY;
	$tileSize = $tileW * $tileH;

	$blockOutY = $blockIdxY * $bs;
	$blockOutXBase = $blockIdxX * $bs;

	for ($i = 0; $i < $k; $i++) {
		$blockOutX = $blockOutXBase + $i * ($gridDimX * $bs);
print("---- ITERACJA K: $i, blockOutX: $blockOutX <br>");
		// Kolektywne ładowanie tile do shared
		for ($idx = $tid; $idx < $tileSize; $idx += $nThreads) {
			$sy = (int)($idx / $tileW);
			$sx = (int)($idx - $sy * $tileW);

            print("&nbsp;&nbsp;&nbsp;&nbsp;WĄTEK: ($threadIdxY, $threadIdxX), IDX: $idx, SY: $sy, SX: $sx, $ii<br>");

			$inY = (int)($blockOutY + $sy);
			$inX = (int)($blockOutX + $sx);
print("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;IN_Y: $inY, IN_X: $inX<br>");

			if ($inY < $N && $inX < $N) $CACHE[$sy * $tileW + $sx] = $IN[$inY * $N + $inX] . '(' . $threadIdxY . ',' . $threadIdxX . ',' . $ii . ')';
			else $CACHE[$sy * $tileW + $sx] = '--';
			$ii++;
		}
	}

	// print("<hr>");
	// print("BLOCK_ID: $blockIdxY, $blockIdxX; THREAD_ID: $threadIdxY, $threadIdxX<br>");

	// print("outSize: $outSize,");
	// print("bs: $bs,");
	// print("tileW: $tileW,");
	// print("tileH: $tileH,");
	// print("tid: $tid<br>");
	// print("nThreads: $nThreads,");
	// print("tileSize: $tileSize,");
	// print("blockOutY: $blockOutY,");
	// print("blockOutXBase: $blockOutXBase,<BR>");
}


function wyswietlTablice($tablica, $szerokosc)
{
	if ($szerokosc <= 0) {
		echo "Szerokość wiersza musi być większa od 0";
		return;
	}

	echo "<table border='1' cellpadding='5' cellspacing='0'>";

	foreach ($tablica as $i => $wartosc) {
		if ($i % $szerokosc == 0) {
			echo "<tr>";
		}

		echo "<td>" . htmlspecialchars($wartosc) . "</td>";

		if (($i + 1) % $szerokosc == 0) {
			echo "</tr>";
		}
	}

	// domknięcie ostatniego wiersza, jeśli niepełny
	if (count($tablica) % $szerokosc != 0) {
		echo "</tr>";
	}

	echo "</table>";
}


$IN = array(
	'A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9',
	'B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9',
	'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
	'D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9',
	'E0', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9',
	'F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9',
	'G0', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9',
	'H0', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9',
	'I0', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9',
	'J0', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9',
);

//wyswietlTablice($IN, 6);

$OUT = array(
	'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX',
	'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX',
	'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX',
	'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX',
	'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX',
	'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX',
	'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX',
	'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX',
);

$CACHE = array(
	'CC', 'CC', 'CC', 'CC', 'CC', 'CC', 'CC', 'CC', 'CC', 'CC',
	'CC', 'CC', 'CC', 'CC', 'CC', 'CC', 'CC', 'CC', 'CC', 'CC',
	'CC', 'CC', 'CC', 'CC', 'CC', 'CC', 'CC', 'CC', 'CC', 'CC',
	'CC', 'CC', 'CC', 'CC', 'CC', 'CC', 'CC', 'CC', 'CC', 'CC',
	'CC', 'CC', 'CC', 'CC', 'CC', 'CC', 'CC', 'CC', 'CC', 'CC',
	'CC', 'CC', 'CC', 'CC', 'CC', 'CC', 'CC', 'CC', 'CC', 'CC',
);


$blockDimX = 4;
$blockDimY = 4;

$threadIdxX = 0;
$threadIdxY = 0;

$gridDimX = 1;
$gridDimY = 2;

$blockIdxY = 0;
$blockIdxX = 0;

// lecimy po blokach
for ($blockIdxY = 0; $blockIdxY < $gridDimY; $blockIdxY++) {
	for ($blockIdxX = 0; $blockIdxX < $gridDimX; $blockIdxX++) {
		print("BLOK ($blockIdxY, $blockIdxX) <br>");
		// lecimy po wątkach
		for ($threadIdxY = 0; $threadIdxY < $blockDimY; $threadIdxY++) {
			for ($threadIdxX = 0; $threadIdxX < $blockDimX; $threadIdxX++) {
				print("&nbsp;&nbsp;WĄTEK ($threadIdxY, $threadIdxX) <br>");
				kernelC($IN, $OUT, 10, 1, 2);
				wyswietlTablice($CACHE, 10);
			}
			//wyswietlTablice($CACHE, 6);
		}
		//$CACHE = array('CC', 'CC', 'CC', 'CC', 'CC', 'CC', 'CC', 'CC', 'CC', 'CC', 'CC', 'CC',	'CC', 'CC', 'CC', 'CC', 'CC', 'CC',	'CC', 'CC', 'CC', 'CC', 'CC', 'CC',	'CC', 'CC', 'CC', 'CC', 'CC', 'CC',					'CC', 'CC', 'CC', 'CC', 'CC', 'CC');
	}
}

wyswietlTablice($IN, 10);
wyswietlTablice($OUT, 8);
wyswietlTablice($CACHE, 6);
