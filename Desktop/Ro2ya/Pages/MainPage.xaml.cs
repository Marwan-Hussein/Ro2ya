using Ro2ya.Services;

namespace Ro2ya.Pages
{
    public partial class MainPage : ContentPage
    {
        private static LocalWebServer? _server;

        public MainPage()
        {
            InitializeComponent();
            EnsureServerStarted();
        }

        public MainPage(PageModels.MainPageModel model) : this()
        {
            BindingContext = model;
        }

        private void EnsureServerStarted()
        {
            if (_server == null)
            {
                _server = new LocalWebServer();
                _server.Start();
            }
            MainWebView.Source = _server.BaseUrl;
        }
    }
}
