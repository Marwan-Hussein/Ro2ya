# 👁️ Ro2ya

**Ro2ya** is an AI-powered desktop application that enables users to control their computer using hand gestures, providing a touchless and more accessible way to interact with the operating system.

The system leverages computer vision and machine learning to recognize predefined gestures in real time and translate them into operating system actions such as cursor movement, volume control, zooming, scrolling, media control, sleep mode, and other productivity features.

Designed as a **Software as a Service (SaaS)** platform, Ro2ya combines local AI inference for fast response times with cloud services for authentication, subscription management, user synchronization, and future feature updates.

---

## 🏗️ High-Level Architecture

Ro2ya consists of four main components:

- **Desktop Application** – Provides the user interface, communicates with the AI engine, executes OS commands, and manages local settings.
- **AI Layer** – Python-based FastAPI service responsible for real-time gesture detection and recognition.
- **Backend API** – Manages authentication, subscriptions, user accounts, licensing, and cloud services.
- **Databases**
  - **SQLite** for local application data and offline settings.
  - **SQL Server / MySQL** for cloud user data, subscriptions, and application services.

---

## 🛠️ Tech Stack

### Desktop Application
- **Framework:** .NET MAUI

### AI Layer
- **Framework:** Python FastAPI
- **Computer Vision:** MediaPipe
- **Machine Learning:** PyTorch

### Backend
- **Framework:** ASP.NET Core Web API
- **Architecture:** Clean Architecture
- **ORM:** Entity Framework Core

### Databases
- **Local Database:** SQLite
- **Cloud Database:** SQL Server / MySQL

### Authentication
- JWT Authentication
- Google OAuth

---

## 🎯 Project Goals

- AI-powered real-time hand gesture recognition.
- Touchless interaction with the operating system.
- Fast and responsive desktop experience through local AI processing.
- Secure cloud-based user accounts and subscription management.
- Modular and scalable architecture that supports future gesture models and operating system features.

---

## 🚀 Getting Started

### Prerequisites

- .NET 8 SDK
- SQL Server

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Marwan-Hussein/ro2ya
   cd Ro2ya
   ```

<!-- 2. **Backend Setup**
   - Restore dependencies for all projects:
     ```bash
     dotnet restore Domain/Domain.csproj
     dotnet restore Application/Application.csproj
     dotnet restore Infrastructure/Infrastructure.csproj
     dotnet restore Forsa/Forsa.csproj
     ```
   - Update `Forsa/appsettings.json` with your connection strings, JWT settings, Google OAuth, Google Calendar, PayMob, and LLM credentials.
   - Run EF Core migrations:
     ```bash
     dotnet ef database update
     ```
   - Run the API:
     ```bash
     cd Forsa
     dotnet run
     ```

3. **Frontend Setup**
   ```bash
   cd Frontend
   npm install
   npm run dev
   ``` -->

---

## 🤝 Contributions

|                                                                                                                                  Contributor                                                                                                                                   |       Role        | Tasks and Lifecycles                |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------: | ----------------------------------- |
|                  <a href="https://github.com/Marwan-Hussein"><img src="https://github.com/Marwan-Hussein.png" width="60px;" alt="Marwan Hussein"/><br /><sub><b>Marwan Hussein</b></sub></a> <p><b>[LinkedIn](https://linkedin.com/in/marwanhussein9)</b></p> | Software Engineer | - Taks1<br> - Task2<br> - Task3<br> |
|                  <a href="https://github.com/mohamed-afifi1"><img src="https://github.com/mohamed-afifi1.png" width="60px;" alt="Marwan Hussein"/><br /><sub><b>Mohamed Afifi</b></sub></a> <p><b>[LinkedIn](https://www.linkedin.com/in/mohammed-afifi-578a03248)</b></p> | Software Engineer | - Taks1<br> - Task2<br> - Task3<br> |
|<a href="https://github.com/AdelHefny"><img src="https://github.com/AdelHefny.png" width="60px;" alt="Marwan Hussein"/><br /><sub><b>Adel Hefny</b></sub></a> <p><b>[LinkedIn](https://www.linkedin.com/in/adel-hefny)</b></p>                         |    AI Engineer    | - Taks1<br> - Task2<br> - Task3<br> |
|<a href="https://github.com/Youssef-Bahaa"><img src="https://github.com/Youssef-Bahaa.png" width="60px;" alt="Marwan Hussein"/><br /><sub><b>Youssef Bahaa</b></sub></a> <p><b>[LinkedIn](https://www.linkedin.com/in/youssef-bahaa-abdelwahab)</b></p>             |    AI Engineer    | - Taks1<br> - Task2<br> - Task3<br> |
|<a href="https://github.com/DevilMO05"><img src="https://github.com/DevilMO05.png" width="60px;" alt="Marwan Hussein"/><br /><sub><b>Mohamed Mahmoud</b></sub></a> <p><b>[LinkedIn](https://www.linkedin.com/in/mohamed-mahmoud-72a1a6310)</b></p>               |    AI Engineer    | - Taks1<br> - Task2<br> - Task3<br> |
| <a href="https://github.com/Mahmoudabdelaziz-2004"><img src="https://github.com/Mahmoudabdelaziz-2004.png" width="60px;" alt="Marwan Hussein"/><br /><sub><b>Mahmoud Abdelaziz</b></sub></a> <p><b>[LinkedIn](https://www.linkedin.com/in/mahmoud-abdelaziz-240012347)</b></p> |  Data Sceintist   | - Taks1<br> - Task2<br> - Task3<br> |

---

## 📜 License

This project is licensed under the LGPL-2.1 License.
