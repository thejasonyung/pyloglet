import tornado.ioloop
import tornado.web
import tornado.escape

from pyloglet import logistic
import json

class TestHandler(tornado.web.RequestHandler):
    def get(self):
        xdata = [ 0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84 ]
        ydata = [ 0.00, 17.93, 36.36, 67.76, 98.10, 131.00, 169.50, 205.50, 228.30, 247.10, 250.50, 253.80, 254.50 ]
        # initial guesses
        p0 = [ 257, 42, 38 ]

        popts = logistic.levenberg_marquardt(p0).fit(xdata, ydata)
        self.write({'params': popts.tolist()})

class LevenbergMarquardtHandler(tornado.web.RequestHandler):
    def post(self):
        """
        curl --data "{\"x\": [ 0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84 ], \"y\": [ 0.00, 17.93, 36.36, 67.76, 98.10, 131.00, 169.50, 205.50, 228.30, 247.10, 250.50, 253.80, 254.50 ], \"init_params\": [ 257, 42, 38 ]}" http://localhost:8888/fit/lm
        """
        print self.request.body
        data = tornado.escape.json_decode(tornado.escape.url_decode(self.request.body))
        popts = logistic.levenberg_marquardt(data['init_params']).fit(data['x'], data['y'])
        self.write({'params': popts.tolist()})


application = tornado.web.Application([
    (r"/", TestHandler),
    (r"/fit/lm", LevenbergMarquardtHandler),
])

if __name__ == "__main__":
    application.listen(8888)
    tornado.ioloop.IOLoop.current().start()
