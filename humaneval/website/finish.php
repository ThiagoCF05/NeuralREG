<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">

<!-- Latest compiled and minified CSS -->
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.4/css/bootstrap.min.css">

<!-- Optional theme -->
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.4/css/bootstrap-theme.min.css">

<!-- Latest compiled and minified JavaScript -->
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.4/js/bootstrap.min.js"></script>

<title>Thank you!</title>
</head>

<body class="container">
    
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

function get_code(){
    $list = $_SESSION["list_id"];
    $conn = connect();
    
    $sql = "SELECT code FROM contexts_lists WHERE id = '$list'";
    $result = $conn->query($sql);
    $row = $result->fetch_assoc();
    $conn->close();
    return $row["code"];
}

function isComplete(){
    $participant_id = $_SESSION["participant_id"];
//     $participant_id = 1;
    $conn = connect();
    
    $sql = "SELECT COUNT(DISTINCT context_id) AS count FROM `experiment5_grades` WHERE participant_id = '$participant_id'";
    $result = $conn->query($sql);
    $row = $result->fetch_assoc();
    $conn->close();
    if ($row["count"] == 24){
        return true;
    } else {
        return false;
    }
}
?>
<article class="text-center">
    <?php 
        session_start();
        if (isComplete() === true) {
            echo "<p class=\"lead\">Well done! You finished the task. Thank you for your collaboration.</p>";
            echo "<p class=\"lead\">For receive your payment, go back to CrowdFlower website and provide the following code:</p>";
            echo "<h1>" .get_code(). "</h1>"; 
        } else {
            echo "<h1>You did not complete the task. Try again.</h1>";
        }
        session_destroy();
    ?>
</article>
</body>
</html>