// 用来计数
const picnum = 5;
// $用来定义JQuery,jQuery 语法是通过选取 HTML 元素，并对选取的元素执行某些操作
// 所有 jQuery 函数位于一个 document ready 函数中，为了防止文档在完全加载（就绪）之前运行 jQuery 代码，即在 DOM 加载完成后才可以对 DOM 进行操作
$(document).ready(function(){
	// 绑定样式
	const slider = $(".slider");
	// 将含有thumbnai样式的div全部取出 以数组的形式装在thumbnails中
	const thumbnails = $(".thumbnail");
	const prevButton = $(".prev-button");
	const nextButton = $(".next-button");
	const slideWidth = slider.width()/picnum;
	let currentSlide = 0;
 
	// 缩略图对应大图的样式
	function setActiveThumbnail(){
		// 清除thumbnail 样式中含有active的样式
		thumbnails.removeClass("active");
		/* eq(),jQuery中的遍历函数 返回带有被选元素的指定索引号的元素
		   这里是向索引为currentSlider中加入active样式*/
		thumbnails.eq(currentSlide).addClass("active");
	}
	
	function slideToSlide(slide){+
		/*css() 方法设置或返回被选元素的一个或多个样式属性
		  translateX 横向平移 向右移动正数 向左移动负数*/
		slider.css("transform","translateX(-"+slide * slideWidth+"px)");
		currentSlide = slide;
		// 调用函数
		setActiveThumbnail();
	}
	
	function nextSlide(){
		// 判断是否位于最后一张图
		if (currentSlide === thumbnails.length -1){
			slideToSlide(0);
		}else{
			slideToSlide(currentSlide + 1);
		}
	}
	
	function prevSlide(){
		// 判断是否位于第一张图
		if(currentSlide === 0){
			// 最后一张图
			slideToSlide(thumbnails.length - 1);
		}else{
			slideToSlide(currentSlide - 1);
		}
	}
	
	nextButton.on("click",nextSlide);
	prevButton.on("click",prevSlide);
	thumbnails.on("click",function(){
		slideToSlide(thumbnails.index(this));
	});
	
	setInterval(nextSlide,2000);
});