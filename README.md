# Analisis numerico - Juliana Hernandez Montoya

πewton is a web page designed to solve numerical methods.

The methods that πewton solves are:
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
   git clone git@github.com:juliana-hernandez/Analisis_numerico.git
   cd Analisis_numerico
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements/development.txt
   ```

4. **Create environment variables**
   - Modify the `.env` file with the appropriate configurations as needed.

5. **Run the server**
   ```bash
   python manage.py runserver
   ```

6. **Access the application**
   - Open your browser and visit: [http://127.0.0.1:8000/]
  
## Notes
- This project does not execute migrations as it does not use a database.
- To add additional features, follow Django's structure for views, templates, and URLs.
