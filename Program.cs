namespace WildLifeAI
{
    internal static class Program
    {
        /// <summary>
        ///  The main entry point for the application.
        /// </summary>
        [STAThread]
        static void Main()
        {
            // To customize application configuration such as set high DPI settings or default font,
            // see https://aka.ms/applicationconfiguration.
            ApplicationConfiguration.Initialize();
            try
            {
                Application.Run(new Form1());
            }
            catch (InvalidOperationException ex) when (ex.Message.StartsWith("Configuration value"))
            {
                MessageBox.Show(ex.Message, "WildLifeAI configuration", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }
    }
}