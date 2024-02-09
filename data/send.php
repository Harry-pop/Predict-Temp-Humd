<?php 

$s = $_GET['s'];
$at = $_GET['at'];
$ah = $_GET['ah'];
$gt = $_GET['gt'];
$gh = $_GET['gh'];

$myFile = "data.txt";
$fh = fopen($myFile, 'a') or die("can't open file");
$stringData = $s . ',' . $at . ',' . $ah . ',' . $gt . ',' . $gh;
fwrite($fh, $stringData."\n");
fclose($fh);

?>