# XNum

XNum is a web application designed to solve numerical methods. It is a simple application with a great user interface and is responsive for use on any device.

The methods that our application solves are:
- Bisection Method
- False Position Method
- Fixed Point Method
- Newton-Raphson Method
- Secant Method
- Multiple Roots Method #1
- Multiple Roots Method #2
- Jacobi's Method
- Gauss-Seidel Method
- SOR Method (Successive Over-Relaxation)
- Vandermonde Method
- Newton Interpolation Method
- Lagrange Method
- Linear and Cubic Spline Methods

## Prerequisites
- Python 3.x installed.
- `pip` installed.
- Virtualenv (optional but recommended).

## Steps to Set Up the Project

1. **Clone the repository**
   ```bash
   git clone git@github.com:kristianrpo/XNum.git
   cd XNum
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements/development.txt
   ```

4. **Create environment variables**
   - Copy the `.env.example` file to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Modify the `.env` file with the appropriate configurations as needed.

5. **Run the server**
   ```bash
   python manage.py runserver
   ```

6. **Access the application**
   - Open your browser and visit: [http://127.0.0.1:8000/](http://127.0.0.1:8000/).
## Notes
- This project does not execute migrations as it does not use a database.
- To add additional features, follow Django's structure for views, templates, and URLs.
