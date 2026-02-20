    def draw_hud(self, img, filename):
        h, w = img.shape[:2]
        overlay = img.copy()

        cv2.rectangle(overlay, (0, 0), (w, 36), (10, 10, 15), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "AI CRASH DETECTION", (10, 24), font, 0.55, (0, 212, 255), 1, cv2.LINE_AA)

        ts = datetime.now().strftime("%H:%M:%S")
        cv2.putText(img, ts, (w - 80, 24), font, 0.5, (100, 100, 130), 1, cv2.LINE_AA)

        cv2.rectangle(overlay, (0, h - 32), (w, h), (10, 10, 15), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)

        cv2.putText(img, filename, (10, h - 10), font, 0.45, (150, 150, 170), 1, cv2.LINE_AA)

        count_text = f"Detections: {self.accident_count}  |  Total: {self.global_accident_count}"
        (tw, _), _ = cv2.getTextSize(count_text, font, 0.45, 1)
        cv2.putText(img, count_text, (w - tw - 10, h - 10), font, 0.45, (255, 58, 92), 1, cv2.LINE_AA)

        if self.accident_count > 0:
            cv2.circle(img, (w - tw - 24, h - 14), 4, (255, 58, 92), -1)

    def display_frame(self, rgb_img):
        if rgb_img is None:
            return

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            self.root.after(50, lambda: self.display_frame(rgb_img))
            return

        ih, iw = rgb_img.shape[:2]
        scale = min(cw / iw, ch / ih)
        nw, nh = int(iw * scale), int(ih * scale)

        resized = cv2.resize(rgb_img, (nw, nh), interpolation=cv2.INTER_AREA)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(resized))

        self.canvas.delete("all")
        self.canvas.create_image((cw - nw) // 2, (ch - nh) // 2, anchor="nw", image=self.photo)

    def log_accident(self, excel_file, file_type, file_name):
        try:
            if not os.path.exists(excel_file):
                wb = Workbook()
                ws = wb.active
                ws.title = "Accident Log"
                ws.append(["Date", "File", "Type", "File Count", "Global Count"])
                wb.save(excel_file)

            wb = load_workbook(excel_file)
            ws = wb.active
            ws.append([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                       file_name, file_type, self.accident_count, self.global_accident_count])
            wb.save(excel_file)
            self.save_global_count()
        except Exception as e:
            self.root.after(0, lambda: self.status_label.configure(text=f"Log error: {str(e)[:40]}"))


def main():
    root = tk.Tk()

    try:
        root.iconbitmap("")
    except:
        pass

    app = ModernAccidentGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
