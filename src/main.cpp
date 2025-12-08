#include "infer/sam3infer.hpp"
#include "common/timer.hpp"
#include "common/cpm.hpp"
#include "osd/osd.hpp"
#include <opencv2/opencv.hpp>
#include <numeric>
#include <thread>
#include <vector>

// ... (全局配置和 setup_data 保持不变) ...
const std::string VISION_MODEL = "model/vision-encoder.engine";
const std::string TEXT_MODEL = "model/text-encoder.engine";
const std::string DECODER_MODEL = "model/decoder.engine";
const std::string GEOMETRY_ENCODER_PATH = "";
const int GPU_ID = 0;

const std::array<int64_t, 32> PERSON_IDS = {49406, 2533, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407};
const std::array<int64_t, 32> PERSON_MASK = {1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

void setup_data(std::shared_ptr<InferBase> engine)
{
    if (engine)
        engine->setup_text_inputs("person", PERSON_IDS, PERSON_MASK);
}

// ... (test_sync 保持不变) ...
void test_sync()
{
    printf("\n=== Sync Test ===\n");
    auto engine = load(VISION_MODEL, TEXT_MODEL, GEOMETRY_ENCODER_PATH, DECODER_MODEL, GPU_ID);
    if (!engine)
        return;
    setup_data(engine);

    cv::Mat img = cv::imread("images/persons.jpg");
    // 强制 Batch = 4 以便与 CPM 的 max_batch 设定公平对比
    std::vector<Sam3Input> inputs;
    for (int i = 0; i < 4; ++i)
        inputs.emplace_back(img, "person");

    engine->forwards(inputs); // Warmup

    nv::EventTimer timer;
    timer.start();
    // 跑 50 次，每次 4 张，总共 200 张
    for (int i = 0; i < 50; ++i)
        engine->forwards(inputs);
    float ms = timer.stop();

    int total_imgs = 50 * 4;
    printf("Sync (Batch=4) Avg Latency: %.2f ms, FPS: %.2f\n", ms / 50.0f, total_imgs / (ms / 1000.0f));
}

using Sam3CPM = cpm::Instance<InferResult, Sam3Input, InferBase>;

// 场景 1: 批量提交 (减少锁竞争，模拟数据积压时的处理能力)
void test_cpm_batch_commit()
{
    printf("\n=== CPM Test (Batch Commit) ===\n");
    Sam3CPM cpm_inst;
    auto loader = []()
    {
        auto eng = load(VISION_MODEL, TEXT_MODEL, GEOMETRY_ENCODER_PATH, DECODER_MODEL, GPU_ID);
        setup_data(eng);
        return eng;
    };

    // Max Batch Size = 4
    if (!cpm_inst.start(loader, 4))
        return;

    cv::Mat img = cv::imread("images/persons.jpg");
    // 准备 200 个请求
    std::vector<Sam3Input> reqs(200, Sam3Input(img, "person"));

    cpm_inst.commit(reqs[0]).get(); // Warmup

    nv::EventTimer timer;
    timer.start();

    // *** 关键修改: 使用 commits 一次性提交所有请求 ***
    // 这会将 200 个请求放入队列，CPM 内部会自动按 Batch=4 切分处理
    // 只会有一次加锁开销
    auto futures = cpm_inst.commits(reqs);

    for (auto &f : futures)
        f.get();

    float ms = timer.stop();
    printf("CPM (Batch Commit) Processed %zu items in %.2f ms. FPS: %.2f\n", reqs.size(), ms, reqs.size() / (ms / 1000.0f));
}

// 场景 2: 多线程并发提交 (模拟真实服务器环境)
void test_cpm_multithread()
{
    printf("\n=== CPM Test (Multi-thread 4 workers) ===\n");
    Sam3CPM cpm_inst;
    auto loader = []()
    {
        auto eng = load(VISION_MODEL, TEXT_MODEL, GEOMETRY_ENCODER_PATH, DECODER_MODEL, GPU_ID);
        setup_data(eng);
        return eng;
    };

    if (!cpm_inst.start(loader, 4))
        return; // Max Batch = 4

    cv::Mat img = cv::imread("images/persons.jpg");
    cpm_inst.commit(Sam3Input(img, "person")).get(); // Warmup

    int num_threads = 4;
    int reqs_per_thread = 50; // 总共 200
    std::vector<std::thread> threads;

    nv::EventTimer timer;
    timer.start();

    for (int t = 0; t < num_threads; ++t)
    {
        threads.emplace_back([&, t]()
                             {
            // 每个线程模拟单独的客户端
            for(int i=0; i<reqs_per_thread; ++i) {
                // 模拟一点点间隔，让 CPM 有机会做 Batching，而不是变成纯粹的串行
                // std::this_thread::sleep_for(std::chrono::microseconds(100)); 
                auto f = cpm_inst.commit(Sam3Input(img, "person"));
                f.get();
            } });
    }

    for (auto &t : threads)
        t.join();

    float ms = timer.stop();
    int total = num_threads * reqs_per_thread;
    printf("CPM (Multi-thread) Processed %d items in %.2f ms. FPS: %.2f\n", total, ms, total / (ms / 1000.0f));
}

int main()
{
    // 对比基准：Sync 使用 Batch=4
    test_sync();

    // 优化1：批量 Commit，消除锁开销，测试 CPM 吞吐极限
    test_cpm_batch_commit();

    // 优化2：多线程提交，模拟真实业务场景
    test_cpm_multithread();

    return 0;
}