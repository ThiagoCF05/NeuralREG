<html>
<body>
    <?php 
        function get_first_page($list){
            if ($list == '1'){ 
                $url = 'list1/20063.php';
            } 
            else if ($list == '2'){
                $url = 'list2/18940.php';
            } 
            else if ($list == '3'){
                $url = 'list3/18844.php';
            } 
            else if ($list == '4'){
                $url = 'list4/18940.php';
            } 
            else if ($list == '5'){
                $url = 'list5/20207.php';
            } 
            else {
                $url = 'list6/19273.php';
            }
            
            return $url;
        }

        function get_ip(){
            if (!empty($_SERVER['HTTP_CLIENT_IP']))   //check ip from share internet
            {
              $ip=$_SERVER['HTTP_CLIENT_IP'];
            }
            elseif (!empty($_SERVER['HTTP_X_FORWARDED_FOR']))   //to check ip is pass from proxy
            {
              $ip=$_SERVER['HTTP_X_FORWARDED_FOR'];
            }
            else
            {
              $ip=$_SERVER['REMOTE_ADDR'];
            }     
            return $ip;
        }

        $servername = "localhost";
        $username = "EZOi4P93";
        $password = "TgPRsGjajDkjl6cQ";
        $dbname = "D2jc45cE";

        $conn = new mysqli($servername, $username, $password, $dbname);
        // Check connection
        if ($conn->connect_error) {
            echo "Fail completely";
            die("Connection failed: " . $conn->connect_error);
        } 

        $name = mysqli_real_escape_string($conn, htmlspecialchars(stripslashes(trim($_POST["name"]))));
        $age = mysqli_real_escape_string($conn, $_POST["age"]);
        $gender = mysqli_real_escape_string($conn, $_POST["gender"]);
        $country = mysqli_real_escape_string($conn, htmlspecialchars(stripslashes(trim($_POST["country"]))));
        $native_language = mysqli_real_escape_string($conn, htmlspecialchars(stripslashes(trim($_POST["native_language"]))));
        $english_proficiency_level = mysqli_real_escape_string($conn, $_POST["english_proficiency_level"]);
        $list = "1";

        // selecting the list with the smaller number of participants
        $sql = "SELECT number_participants FROM contexts_lists WHERE id = '$list'";
        $participant = 0;
        $result = $conn->query($sql);
        if ($result->num_rows > 0) {
            // output data of each row
            while($row = $result->fetch_assoc()) {
                $participant = $row["number_participants"];
            }
        }

        $sql = "SELECT `id`, `number_participants` FROM `contexts_lists` WHERE `number_participants` IN (SELECT MIN(`number_participants`) FROM `contexts_lists` WHERE id IN (1,2,3,4,5,6))";
        $participant = 0;
        $result = $conn->query($sql);
        $row = $result->fetch_assoc();
        $participant = $row["number_participants"];
        $list = $row["id"];

        // Increasing the number of participants in the chosen list
        $participant = $participant + 1;
        $sql = "UPDATE contexts_lists SET number_participants = '$participant' WHERE id = '$list'";
        if ($conn->query($sql) === TRUE) {
            # get IP of the user
            $ip = get_ip();
            $sql = "INSERT INTO experiment5_participants (name, age, gender, country, native_language, english_proficiency_level, ip_address, list_id) VALUES ('$name', '$age', '$gender', '$country', '$native_language', '$english_proficiency_level', '$ip', '$list')";
            if ($conn->query($sql) === TRUE) {
                session_start();
                $sql = "SELECT id FROM experiment5_participants WHERE name = '$name' AND age = '$age' AND gender = '$gender ' AND list_id = '$list'";
                $result = $conn->query($sql);
                if ($result->num_rows > 0) {
//                     // output data of each row
                    while($row = $result->fetch_assoc()) {
                        $participant_id = $row["id"];
                    }
                    $_SESSION["participant_id"] = $participant_id;
                    $_SESSION["list_id"] = $list;
                    $_SESSION["name"] = $name;
                    $_SESSION["gender"] = $gender;
                    $_SESSION["age"] = $age;
                    $page = get_first_page($list, $conn);
                    header("Location: $page");
                    die();
                }
            } 
        }
        // BEGIN TEST VERSION
        // $page = "list" . $list . "/" . "l1c1-i10o-spain.php";
        // session_start();
        // $_SESSION["participant_id"] = '1';
        // $_SESSION["list_id"] = $list;
        // header("Location: $page");
        // die();
        // END TEST VERSION

        $conn->close();
    ?>
</body>
</html>
