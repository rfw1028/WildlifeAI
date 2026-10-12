namespace WildLifeAI
{
    partial class Form1
    {

        private void InitializeComponent()
        {
            button1 = new Button();
            button2 = new Button();
            rtb1 = new RichTextBox();
            SuspendLayout();
            // 
            // button1
            // 
            button1.Location = new Point(1833, 370);
            button1.Name = "button1";
            button1.Size = new Size(89, 27);
            button1.TabIndex = 1;
            button1.Text = "START";
            button1.UseVisualStyleBackColor = true;
            button1.Click += button1_Click;
            // 
            // button2
            // 
            button2.Location = new Point(1833, 440);
            button2.Name = "button2";
            button2.Size = new Size(89, 27);
            button2.TabIndex = 2;
            button2.Text = "STOP";
            button2.UseVisualStyleBackColor = true;
            // 
            // rtb1
            // 
            rtb1.Location = new Point(1721, 14);
            rtb1.Name = "rtb1";
            rtb1.Size = new Size(308, 326);
            rtb1.TabIndex = 3;
            rtb1.Text = "";
            // 
            // Form1
            // 
            AutoScaleDimensions = new SizeF(8F, 20F);
            AutoScaleMode = AutoScaleMode.Font;
            AutoSize = true;
            AutoSizeMode = AutoSizeMode.GrowAndShrink;
            ClientSize = new Size(1296, 844);
            Controls.Add(rtb1);
            Controls.Add(button2);
            Controls.Add(button1);
            Margin = new Padding(3, 4, 3, 4);
            Name = "Form1";
            Text = "WildLifeAI";
            Load += Form1_Load;
            ResumeLayout(false);
        }
        private System.ComponentModel.IContainer components;
        private Button button1;
        private Button button2;
        private RichTextBox rtb1;
    }
}
