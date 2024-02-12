// Motor driver 1 pins
const int motor1Pin1 = 2;
const int motor1Pin2 = 3;
const int motor2Pin1 = 4;
const int motor2Pin2 = 5;

// Motor driver 2 pins
const int motor3Pin1 = 6;
const int motor3Pin2 = 7;
const int motor4Pin1 = 8;
const int motor4Pin2 = 9;

// Serial input
const int serialInputPin = 10;

void setup() {
  // Motor driver 1 pins as outputs
  pinMode(motor1Pin1, OUTPUT);
  pinMode(motor1Pin2, OUTPUT);
  pinMode(motor2Pin1, OUTPUT);
  pinMode(motor2Pin2, OUTPUT);

  // Motor driver 2 pins as outputs
  pinMode(motor3Pin1, OUTPUT);
  pinMode(motor3Pin2, OUTPUT);
  pinMode(motor4Pin1, OUTPUT);
  pinMode(motor4Pin2, OUTPUT);

  // Serial communication
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    char serialInput = Serial.read(); // Read a single character from the serial input
    
    // Perform actions based on serial input
    if (serialInput == '0') {
      // Move the car
      stopCar();
    } else if (serialInput == 'c') {
      // Stop the car
      moveCar();
    } else if (serialInput == 'l') {
      // Stop the car
      turnLeft();
    } else if (serialInput == 'r') {
      // Stop the car
      turnRight();
    }
  }
  
  // Check for directional commands
//  if (Serial.available() > 0) {
//    String direction = Serial.readString(); // Read the direction from serial input
//    
//    // Adjust car's direction based on the received command
//    if (direction == "left") {
//      turnLeft();
//    } else if (direction == "right") {
//      turnRight();
//    } else if (direction == "center") {
//      moveCar();
//    }
//  }
}

// Function to move the car forward
void moveCar() {
  digitalWrite(motor1Pin1, HIGH);
  digitalWrite(motor1Pin2, LOW);
  
  digitalWrite(motor2Pin1, HIGH);
  digitalWrite(motor2Pin2, LOW);
  
  digitalWrite(motor3Pin1, HIGH);
  digitalWrite(motor3Pin2, LOW);
  
  digitalWrite(motor4Pin1, HIGH);
  digitalWrite(motor4Pin2, LOW);
}

// Function to stop the car
void stopCar() {
  digitalWrite(motor1Pin1, LOW);
  digitalWrite(motor1Pin2, LOW);
  digitalWrite(motor2Pin1, LOW);
  digitalWrite(motor2Pin2, LOW);
  digitalWrite(motor3Pin1, LOW);
  digitalWrite(motor3Pin2, LOW);
  digitalWrite(motor4Pin1, LOW);
  digitalWrite(motor4Pin2, LOW);
}

// Function to turn the car left
void turnRight() {
  digitalWrite(motor1Pin1, LOW);
  digitalWrite(motor1Pin2, HIGH);
  
  digitalWrite(motor2Pin1, HIGH);
  digitalWrite(motor2Pin2, LOW);
  
  digitalWrite(motor3Pin1, LOW);
  digitalWrite(motor3Pin2, HIGH);
  
  digitalWrite(motor4Pin1, HIGH);
  digitalWrite(motor4Pin2, LOW);
}

// Function to turn the car right
void turnLeft() {
  digitalWrite(motor1Pin1, HIGH);
  digitalWrite(motor1Pin2, LOW);
  
  digitalWrite(motor2Pin1, LOW);
  digitalWrite(motor2Pin2, HIGH);
  
  digitalWrite(motor3Pin1, HIGH);
  digitalWrite(motor3Pin2, LOW);
  
  digitalWrite(motor4Pin1, LOW);
  digitalWrite(motor4Pin2, HIGH);
}