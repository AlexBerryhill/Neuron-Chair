// Define motor control pins
#define MOTOR1_PIN1 2
#define MOTOR1_PIN2 3
#define MOTOR2_PIN1 4
#define MOTOR2_PIN2 5
#define MOTOR3_PIN1 6
#define MOTOR3_PIN2 7
#define MOTOR4_PIN1 8
#define MOTOR4_PIN2 9

void setup() {
  // Set all the motor control pins to outputs
  pinMode(MOTOR1_PIN1, OUTPUT);
  pinMode(MOTOR1_PIN2, OUTPUT);
  pinMode(MOTOR2_PIN1, OUTPUT);
  pinMode(MOTOR2_PIN2, OUTPUT);
  pinMode(MOTOR3_PIN1, OUTPUT);
  pinMode(MOTOR3_PIN2, OUTPUT);
  pinMode(MOTOR4_PIN1, OUTPUT);
  pinMode(MOTOR4_PIN2, OUTPUT);

  Serial.begin(9600);  // Start serial communication at 9600 bps
}

void loop() {
  if (Serial.available()) {  // If data is available to read
    char val = Serial.read();  // Read it
    move(val);
  }
}

void move(char command) {
  switch(command) {
    case '0':  // Stop
      // All motors are stopped
      /*
        [Motor1]   [Motor2]
            X         X
             ----------
             |        |
             |  Car   |
             |        |
             ----------
        [Motor3]   [Motor4]
            X         X
      */
      digitalWrite(MOTOR1_PIN1, LOW);
      digitalWrite(MOTOR1_PIN2, LOW);
      digitalWrite(MOTOR2_PIN1, LOW);
      digitalWrite(MOTOR2_PIN2, LOW);
      digitalWrite(MOTOR3_PIN1, LOW);
      digitalWrite(MOTOR3_PIN2, LOW);
      digitalWrite(MOTOR4_PIN1, LOW);
      digitalWrite(MOTOR4_PIN2, LOW);
      break;
    case 'l':  // Left
      // Motors on the left side (Motor1 and Motor3) are moving forward, and motors on the right side (Motor2 and Motor4) are stopped
      /*
        [Motor1]   [Motor2]
            ↑         X
             ----------
             |        |
             |  Car   |
             |        |
             ----------
        [Motor3]   [Motor4]
            ↑         X
      */
      digitalWrite(MOTOR1_PIN1, HIGH);
      digitalWrite(MOTOR1_PIN2, LOW);
      digitalWrite(MOTOR2_PIN1, LOW);
      digitalWrite(MOTOR2_PIN2, HIGH);
      digitalWrite(MOTOR3_PIN1, HIGH);
      digitalWrite(MOTOR3_PIN2, LOW);
      digitalWrite(MOTOR4_PIN1, LOW);
      digitalWrite(MOTOR4_PIN2, HIGH);
      break;
    case 'r':  // Right
      // Motors on the right side (Motor2 and Motor4) are moving forward, and motors on the left side (Motor1 and Motor3) are stopped
      /*
        [Motor1]   [Motor2]
            X         ↑
             ----------
             |        |
             |  Car   |
             |        |
             ----------
        [Motor3]   [Motor4]
            X         ↑
      */
      digitalWrite(MOTOR1_PIN1, LOW);
      digitalWrite(MOTOR1_PIN2, HIGH);
      digitalWrite(MOTOR2_PIN1, HIGH);
      digitalWrite(MOTOR2_PIN2, LOW);
      digitalWrite(MOTOR3_PIN1, LOW);
      digitalWrite(MOTOR3_PIN2, HIGH);
      digitalWrite(MOTOR4_PIN1, HIGH);
      digitalWrite(MOTOR4_PIN2, LOW);
      break;
    case 'c':  // Straight
      // All motors are moving forward
      /*
        [Motor1]   [Motor2]
            ↑         ↑
             ----------
             |        |
             |  Car   |
             |        |
             ----------
        [Motor3]   [Motor4]
            ↑         ↑
      */
      digitalWrite(MOTOR1_PIN1, HIGH);
      digitalWrite(MOTOR1_PIN2, LOW);
      digitalWrite(MOTOR2_PIN1, HIGH);
      digitalWrite(MOTOR2_PIN2, LOW);
      digitalWrite(MOTOR3_PIN1, HIGH);
      digitalWrite(MOTOR3_PIN2, LOW);
      digitalWrite(MOTOR4_PIN1, HIGH);
      digitalWrite(MOTOR4_PIN2, LOW);
      break;
  }
}
