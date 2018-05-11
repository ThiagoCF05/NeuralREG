<?php
function connect(){
    $servername = "localhost";
    $username = "EZOi4P93";
    $password = "TgPRsGjajDkjl6cQ";
    $dbname = "D2jc45cE";
    
    $conn = new mysqli($servername, $username, $password, $dbname);
    // Check connection
    if ($conn->connect_error) {
        die("Connection failed: " . $conn->connect_error);
    }
    return $conn;
}

function get_url_information($conn, $url){
    $sql = "SELECT id, next_url FROM `experiment5_contexts` WHERE url = '$url'";
    $result = $conn->query($sql);
    $row = $result->fetch_assoc();
    return $row;
}

function insert_grade($conn, $context_id, $participant_id, $fluency, $grammar, $clarity){
    $sql = "INSERT INTO `experiment5_grades` (fluency, grammar, clarity, participant_id, context_id) VALUES ('$fluency', '$grammar', '$clarity', '$participant_id', '$context_id');";
    $conn->query($sql);
}

$conn = connect();
session_start();
$participant_id = $_SESSION["participant_id"];
$url = mysqli_real_escape_string($conn, htmlspecialchars(stripslashes(trim($_POST["url"]))));
$fluency = mysqli_real_escape_string($conn, htmlspecialchars(stripslashes(trim($_POST["fluency"]))));
$grammar = mysqli_real_escape_string($conn, htmlspecialchars(stripslashes(trim($_POST["grammar"]))));
$clarity = mysqli_real_escape_string($conn, htmlspecialchars(stripslashes(trim($_POST["clarity"]))));
$url = "list" .$_SESSION["list_id"]. "/" . $url;
$url_information = get_url_information($conn, $url);
$next_page = $url_information["next_url"];
$page_id = $url_information["id"];
insert_grade($conn, $page_id, $participant_id, $fluency, $grammar, $clarity);
$conn->close();
header("Location: $next_page");
die();
?>