
        var currentImageIndex = 0;
        var images = [];

        // 获取图片路径列表
        function getImagesFromFolder() {
            fetch('/get_images')  // 发送GET请求获取图片路径列表
                .then(response => response.json())
                .then(data => {
                    images = data.images;

                    // 显示总览列表
                    showOverview();

                    // 默认显示第一张图片
                    showImage(currentImageIndex);
                });
        }

        // 显示当前索引对应的图片
        function showImage(index) {
            if (images.length > 0) {
                document.getElementById('image').src = images[index];
            }
        }

        // 切换到上一张图片
        function previousImage() {
            if (currentImageIndex > 0) {
                currentImageIndex--;
                showImage(currentImageIndex);
            }
        }

        // 切换到下一张图片
        function nextImage() {
            if (currentImageIndex < images.length - 1) {
                currentImageIndex++;
                showImage(currentImageIndex);
            }
        }

        // 显示总览列表
        function showOverview() {
            var imageList = document.getElementById("imageList");
            imageList.innerHTML = "";
            for (var i = 0; i < images.length; i++) {
                var listItem = document.createElement("li");
                listItem.textContent = images[i].substring(images[i].lastIndexOf("/") + 1);
                listItem.addEventListener("click", function() {
                    var index = Array.from(imageList.children).indexOf(this);
                    currentImageIndex = index;
                    showImage(currentImageIndex);
                    toggleOverview();
                });
                imageList.appendChild(listItem);
            }
        }

        // 显示或隐藏总览列表
        function toggleOverview() {
            var imageList = document.getElementById("imageList");
            if (imageList.style.display === "none") {
                imageList.style.display = "block";
            } else {
                imageList.style.display = "none";
            }
        }


        // 页面加载完成后调用
        window.onload = function() {
            getImagesFromFolder();
        };
