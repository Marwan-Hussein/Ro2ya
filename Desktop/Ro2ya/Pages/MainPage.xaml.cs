using Ro2ya.Models;
using Ro2ya.PageModels;

namespace Ro2ya.Pages
{
    public partial class MainPage : ContentPage
    {
        public MainPage(MainPageModel model)
        {
            InitializeComponent();
            BindingContext = model;
        }
    }
}