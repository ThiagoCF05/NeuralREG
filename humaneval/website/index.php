<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">

<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-beta/css/bootstrap.min.css" integrity="sha384-/Y6pD6FV/Vv2HJnA6t+vslU6fwYXjCFtcEpHbNJ0lyAFsXTsjBbfaDjzALeQsN6M" crossorigin="anonymous">
<script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.11.0/umd/popper.min.js" integrity="sha384-b/U6ypiBEHpOf/4+1nzFpr53nxSS+GLCkfwBdFNTxtclqqenISfwAzpKaMNFNmj4" crossorigin="anonymous"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-beta/js/bootstrap.min.js" integrity="sha384-h0AbiXch4ZDo7tp9hKZ4TsHbi047NrKGLO3SEJAg45jXxnGIfYzk4Si90RDIqNm1" crossorigin="anonymous"></script>

<title>Tilburg University - TiCC</title>
</head>

<body class="container">
	<?php
        function getBrowser() 
        { 
            $u_agent = $_SERVER['HTTP_USER_AGENT']; 
            $bname = 'Unknown';
            $platform = 'Unknown';
            $version= "";

            //First get the platform?
            if (preg_match('/linux/i', $u_agent)) {
                $platform = 'linux';
            }
            elseif (preg_match('/macintosh|mac os x/i', $u_agent)) {
                $platform = 'mac';
            }
            elseif (preg_match('/windows|win32/i', $u_agent)) {
                $platform = 'windows';
            }

            // Next get the name of the useragent yes seperately and for good reason
            if(preg_match('/MSIE/i',$u_agent) && !preg_match('/Opera/i',$u_agent)) 
            { 
                $bname = 'Internet Explorer'; 
                $ub = "MSIE"; 
            } 
            elseif(preg_match('/Firefox/i',$u_agent)) 
            { 
                $bname = 'Mozilla Firefox'; 
                $ub = "Firefox"; 
            } 
            elseif(preg_match('/Chrome/i',$u_agent)) 
            { 
                $bname = 'Google Chrome'; 
                $ub = "Chrome"; 
            } 
            elseif(preg_match('/Safari/i',$u_agent)) 
            { 
                $bname = 'Apple Safari'; 
                $ub = "Safari"; 
            } 
            elseif(preg_match('/Opera/i',$u_agent)) 
            { 
                $bname = 'Opera'; 
                $ub = "Opera"; 
            } 
            elseif(preg_match('/Netscape/i',$u_agent)) 
            { 
                $bname = 'Netscape'; 
                $ub = "Netscape"; 
            } 

            // finally get the correct version number
            $known = array('Version', $ub, 'other');
            $pattern = '#(?<browser>' . join('|', $known) .
            ')[/ ]+(?<version>[0-9.|a-zA-Z.]*)#';
            if (!preg_match_all($pattern, $u_agent, $matches)) {
                // we have no matching number just continue
            }

            // see how many we have
            $i = count($matches['browser']);
            if ($i != 1) {
                //we will have two since we are not using 'other' argument yet
                //see if version is before or after the name
                if (strripos($u_agent,"Version") < strripos($u_agent,$ub)){
                    $version= $matches['version'][0];
                }
                else {
                    $version= $matches['version'][1];
                }
            }
            else {
                $version= $matches['version'][0];
            }

            // check if we have a number
            if ($version==null || $version=="") {$version="?";}

            return array(
                'userAgent' => $u_agent,
                'name'      => $bname,
                'version'   => $version,
                'platform'  => $platform,
                'pattern'    => $pattern
            );
        }

        function is_participant(){
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
            
            $servername = "localhost";
            $username = "EZOi4P93";
            $password = "TgPRsGjajDkjl6cQ";
            $dbname = "D2jc45cE";

            $conn = new mysqli($servername, $username, $password, $dbname);
            // Check connection
            if ($conn->connect_error) {
                die("Connection failed: " . $conn->connect_error);
            } 

            $sql = "SELECT id FROM experiment5_participants WHERE ip_address = '$ip'";
            $result = $conn->query($sql);
            if ($result->num_rows > 0) {
                return false;
            } else {
                return true;
            }
        }
        $ua=getBrowser();
        $isChrome = strcmp($ua['name'], 'Google Chrome');
        $isFirefox = strcmp($ua['name'], 'Mozilla Firefox');
        if ($isChrome != 0 && $isFirefox != 0){
            header("Location: 404.php");
            die();
        }
        
        if (!is_participant()){
            header("Location: completed.php");
            die();
        }
    ?>
	<div class="row">
		<div class="col-md-12 text-center">
			<img src="img/tilburg-university-logo.jpg">
		</div>
	</div>
	<div class="text-center">
		<p class="lead">
            <strong>Welcome!</strong> Thank you for participating in our research. Please read the instructions carefully.
		</p>
	</div>
	<div class="text-justify">
		<section id="Proceedings">
			<h3>Proceedings</h3>
			<p class="lead">
                In the next pages, you will be presented with 24 very short texts, each describing pieces of data, expressing properties and relations of entities. In the texts, references to entities are highlighted in yellow, as in the following example:
			</p>

			<div class="jumbotron lead">
                <div class="row justify-content-center">
                  <div class="col-6">
                    <h5 class="text-center">Data</h5>
                    <table class="table table-striped small">
                      <tr>
                        <td>Adolfo_Suárez_Madrid–Barajas_Airport</td>
                        <td><strong>runwayLength</strong></td>
                        <td>4349.0</td>
                      </tr>
                      <tr>
                        <td>Adolfo_Suárez_Madrid–Barajas_Airport</td>
                        <td><strong>location</strong></td>
                        <td>Madrid</td>
                      </tr>
                      <tr>
                        <td>Adolfo_Suárez_Madrid–Barajas_Airport</td>
                        <td><strong>elevationAboveTheSeaLevel_(in_metres)</strong></td>
                        <td>610.0</td>
                      </tr>
                      <tr>
                        <td>Adolfo_Suárez_Madrid–Barajas_Airport</td>
                        <td><strong>operatingOrganisation</strong></td>
                        <td>ENAIRE</td>
                      </tr>
                      <tr>
                        <td>Adolfo_Suárez_Madrid–Barajas_Airport</td>
                        <td><strong>runwayName</strong></td>
                        <td>"14L/32R"</td>
                      </tr>
                    </table>
                  </div>
                </div>
                <br>
                <div class="text-justify" id="article">
                  <h5 class="text-center">Summary</h5>
                  <p class="lead">
                    <p class="lead" id="text_article"><span style="background-color: #ffff00">adofo suárez madrid-barajas airport</span> , which lies 610 metres above sea level , is located in <span style="background-color: #ffff00">madrid</span> and operated by <span style="background-color: #ffff00">enaire</span> . <span style="background-color: #ffff00">the airport 's</span> runway , named <span style="background-color: #ffff00">14l/32r</span> , has a length of 4349.0 .</p>
                  </p>
                </div>
			</div>

			<p class="lead">
                We would like to hear your opinion about the quality of the texts to describe the data, taking into account these highlighted references. In particular, we would like you to evaluate the <strong>fluency</strong> (does the text flow in a natural, easy to read manner?), <strong>grammaticality</strong> (is the text grammatical (no spelling or grammatical errors)?) and <strong>clarity</strong> of the texts (does the text clearly express the data?), with special emphasis on the references. 
			</p>

			<p class="lead">
                Please rate these three dimensions on a scale from <strong>Very Bad</strong> to <strong>Very Good</strong>. As you may see by our example, all words in the text are <strong>lowercased</strong> and <strong>tokenized</strong> (all units in the text, including punctuation, are separated by whitespaces). We ask you <strong>to do not take these issues into account in your evaluation</strong>.
            </p>
            <p class="lead">
                The experiment will last around 15-20 minutes. It should be done without pauses. Hence, be sure to start it only if you have that time available.   
            </p>
		</section>
        <section>
            <h3>Payment</h3>
            <p class="lead">
                 At the end of the experiment, a code will be displayed. To receive <strong>1 dollar</strong> for your participation, you must provide that code on the Crowd Flower page that redirected you to here. <strong>Remember to keep that CrowdFlower page opened while you are working on the experiment. If you close it, you will not be able to insert the code, and receive the payment.</strong> 
            </p>
        </section>
		<section id="terms">
			<h3>Terms</h3>
			<p class="lead">
				 Your information will be used for research purposes only. All your data will be treated anonymously. 
			</p>
			<p class="lead">
                If you agree with the information presented above and want to proceed with the experiment, please fill the following form and press the button ‘I agree’. 			
            </p>
			<section class="jumbotron">
				<form class="lead" action="login.php" method="post">
					<div class="form-group row">
						<label for="inputName" class="col-md-2 control-label">Name</label>
						<div class="col-md-6">
							<input type="text" class="form-control" id="inputName" name="name" placeholder="Name" required maxlength="45" minlength="3">
                            <!--<span id="helpBlock" class="help-block">The text should go here.</span>-->
						</div>
					</div>
                    <div class="form-group row">
						<label class="col-md-2 control-label">Sex</label>
						<div class="col-md-3">
							<select class="form-control" name="gender">
                                <option value="M">Male</option>
                                <option value="F">Female</option>
							</select>
						</div>
					</div>
					<div class="form-group row">
						<label class="col-md-2 control-label">Age</label>
						<div class="col-md-2">
							<select class="form-control" name="age">
								<?php 
									for ($x = 18; $x <= 80; $x++) {
    									echo "<option value=\"$x\">$x</option>";
									} 
								?>
							</select>
						</div>
					</div>
					<div class="form-group row">
						<label class="col-md-2 control-label">Country</label>
						<div class="col-md-3">
							<select class="form-control" name="country" required>
								<?php
 
									$countries = array("Afghanistan", "Albania", "Algeria", "American Samoa", "Andorra", "Angola", "Anguilla", "Antarctica", "Antigua and Barbuda", "Argentina", "Armenia", "Aruba", "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bermuda", "Bhutan", "Bolivia", "Bosnia and Herzegowina", "Botswana", "Bouvet Island", "Brazil", "British Indian Ocean Territory", "Brunei Darussalam", "Bulgaria", "Burkina Faso", "Burundi", "Cambodia", "Cameroon", "Canada", "Cape Verde", "Cayman Islands", "Central African Republic", "Chad", "Chile", "China", "Christmas Island", "Cocos (Keeling) Islands", "Colombia", "Comoros", "Congo", "Congo, the Democratic Republic of the", "Cook Islands", "Costa Rica", "Cote d'Ivoire", "Croatia (Hrvatska)", "Cuba", "Cyprus", "Czech Republic", "Denmark", "Djibouti", "Dominica", "Dominican Republic", "East Timor", "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", "Ethiopia", "Falkland Islands (Malvinas)", "Faroe Islands", "Fiji", "Finland", "France", "France Metropolitan", "French Guiana", "French Polynesia", "French Southern Territories", "Gabon", "Gambia", "Georgia", "Germany", "Ghana", "Gibraltar", "Greece", "Greenland", "Grenada", "Guadeloupe", "Guam", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti", "Heard and Mc Donald Islands", "Holy See (Vatican City State)", "Honduras", "Hong Kong", "Hungary", "Iceland", "India", "Indonesia", "Iran (Islamic Republic of)", "Iraq", "Ireland", "Israel", "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Korea, Democratic People's Republic of", "Korea, Republic of", "Kuwait", "Kyrgyzstan", "Lao, People's Democratic Republic", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libyan Arab Jamahiriya", "Liechtenstein", "Lithuania", "Luxembourg", "Macau", "Macedonia, The Former Yugoslav Republic of", "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands", "Martinique", "Mauritania", "Mauritius", "Mayotte", "Mexico", "Micronesia, Federated States of", "Moldova, Republic of", "Monaco", "Mongolia", "Montserrat", "Morocco", "Mozambique", "Myanmar", "Namibia", "Nauru", "Nepal", "Netherlands", "Netherlands Antilles", "New Caledonia", "New Zealand", "Nicaragua", "Niger", "Nigeria", "Niue", "Norfolk Island", "Northern Mariana Islands", "Norway", "Oman", "Pakistan", "Palau", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Pitcairn", "Poland", "Portugal", "Puerto Rico", "Qatar", "Reunion", "Romania", "Russian Federation", "Rwanda", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa", "San Marino", "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Seychelles", "Sierra Leone", "Singapore", "Slovakia (Slovak Republic)", "Slovenia", "Solomon Islands", "Somalia", "South Africa", "South Georgia and the South Sandwich Islands", "Spain", "Sri Lanka", "St. Helena", "St. Pierre and Miquelon", "Sudan", "Suriname", "Svalbard and Jan Mayen Islands", "Swaziland", "Sweden", "Switzerland", "Syrian Arab Republic", "Taiwan, Province of China", "Tajikistan", "Tanzania, United Republic of", "Thailand", "Togo", "Tokelau", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Turks and Caicos Islands", "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom", "United States", "United States Minor Outlying Islands", "Uruguay", "Uzbekistan", "Vanuatu", "Venezuela", "Vietnam", "Virgin Islands (British)", "Virgin Islands (U.S.)", "Wallis and Futuna Islands", "Western Sahara", "Yemen", "Yugoslavia", "Zambia", "Zimbabwe");
									foreach ($countries as $country) {
    									echo "<option value=\"$country\">$country</option>";
									}
								?>
							</select>
						</div>
					</div>
					<div class="form-group row">
						<label class="col-md-2 control-label">Native Language</label>
						<div class="col-md-3">
							<input type="text" class="form-control" name="native_language" placeholder="Native Language" required maxlength="45" minlength="3">
						</div>
					</div>
					<div class="form-group row">
						<label class="col-md-2 control-label">English Proficiency Level</label>
						<div class="col-md-3">
							<select class="form-control" name="english_proficiency_level">
                                <option value="native">Native</option>
                                <option value="fluent">Fluent</option>
                                <option value="basic">Basic</option>
							</select>
						</div>
					</div>
					<div class="form-group row">
						<div class="col-md-3">
							<button type="submit" class="btn btn-primary">I agree</button>
						</div>
					</div>
				</form>
			</section>
		</section>
	</div>
	<footer class="footer">
		<p> Tilburg center for Cognition and Communication - Tilburg University 2018</p>
	</footer>
</body>
</html>